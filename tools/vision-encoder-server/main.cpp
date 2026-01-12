/**
 * Vision Encoder Server
 * 
 * HTTP server that encodes images to embeddings using CLIP vision encoder.
 * Part of the distributed VLM inference architecture.
 * 
 * Usage:
 *   vision-encoder-server -m /path/to/mmproj.gguf [-p 8081]
 * 
 * Endpoints:
 *   GET  /health           - Health check
 *   POST /v1/vision/encode - Encode image to embedding
 */

// We need internal CLIP headers for direct access
#define MTMD_INTERNAL_HEADER
#include "clip.h"
#include "clip-impl.h"
#include "ggml.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <memory>

using json = nlohmann::json;

// Base64 decoding
static const std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

static inline bool is_base64(unsigned char c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
}

static std::vector<unsigned char> base64_decode(const std::string& encoded_string) {
    size_t in_len = encoded_string.size();
    size_t i = 0;
    size_t in_ = 0;
    unsigned char char_array_4[4], char_array_3[3];
    std::vector<unsigned char> ret;
    ret.reserve(in_len * 3 / 4);

    while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
        char_array_4[i++] = encoded_string[in_]; in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++)
                char_array_4[i] = base64_chars.find(char_array_4[i]);

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (i = 0; (i < 3); i++)
                ret.push_back(char_array_3[i]);
            i = 0;
        }
    }

    if (i) {
        for (size_t j = i; j < 4; j++)
            char_array_4[j] = 0;

        for (size_t j = 0; j < 4; j++)
            char_array_4[j] = base64_chars.find(char_array_4[j]);

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

        for (size_t j = 0; (j < i - 1); j++)
            ret.push_back(char_array_3[j]);
    }

    return ret;
}

struct vision_encoder_server {
    clip_ctx * ctx = nullptr;
    std::mutex encode_mutex;  // Protect encoding (not thread-safe)
    int n_threads = 4;
    
    bool init(const char * model_path, bool use_gpu) {
        clip_context_params params;
        params.use_gpu = use_gpu;
        params.flash_attn_type = CLIP_FLASH_ATTN_TYPE_AUTO;
        params.image_min_tokens = 0;
        params.image_max_tokens = 0;
        params.warmup = true;
        
        std::cout << "Loading vision encoder from: " << model_path << std::endl;
        
        auto result = clip_init(model_path, params);
        ctx = result.ctx_v;
        
        if (!ctx) {
            std::cerr << "Failed to load vision model from: " << model_path << std::endl;
            return false;
        }
        
        std::cout << "Vision encoder loaded successfully!" << std::endl;
        std::cout << "  Hidden size: " << clip_get_hidden_size(ctx) << std::endl;
        std::cout << "  Image size: " << clip_get_image_size(ctx) << std::endl;
        std::cout << "  Patch size: " << clip_get_patch_size(ctx) << std::endl;
        std::cout << "  Mmproj embd: " << clip_n_mmproj_embd(ctx) << std::endl;
        
        return true;
    }
    
    ~vision_encoder_server() {
        if (ctx) {
            clip_free(ctx);
        }
    }
    
    json encode_image_base64(const std::string& base64_data) {
        std::lock_guard<std::mutex> lock(encode_mutex);
        
        // Decode base64
        auto image_bytes = base64_decode(base64_data);
        if (image_bytes.empty()) {
            return json{{"error", "Failed to decode base64 image"}};
        }
        
        // Load image using stb_image
        int nx, ny, nc;
        unsigned char * data = stbi_load_from_memory(
            image_bytes.data(), 
            image_bytes.size(), 
            &nx, &ny, &nc, 3  // Force RGB
        );
        
        if (!data) {
            return json{{"error", "Failed to decode image data"}};
        }
        
        std::cout << "Image loaded: " << nx << "x" << ny << " (channels: " << nc << ")" << std::endl;
        
        // Build clip_image_u8 from raw RGB pixels
        clip_image_u8 * img_u8 = clip_image_u8_init();
        clip_build_img_from_pixels(data, nx, ny, img_u8);
        stbi_image_free(data);
        
        // Preprocess image to f32
        clip_image_f32_batch * img_batch = clip_image_f32_batch_init();
        if (!clip_image_preprocess(ctx, img_u8, img_batch)) {
            clip_image_u8_free(img_u8);
            clip_image_f32_batch_free(img_batch);
            return json{{"error", "Failed to preprocess image"}};
        }
        
        clip_image_u8_free(img_u8);
        
        // Get the first preprocessed image
        if (clip_image_f32_batch_n_images(img_batch) == 0) {
            clip_image_f32_batch_free(img_batch);
            return json{{"error", "No images after preprocessing"}};
        }
        
        clip_image_f32 * img_f32 = clip_image_f32_get_img(img_batch, 0);
        
        // Calculate embedding size
        int n_tokens = clip_n_output_tokens(ctx, img_f32);
        int hidden_dim = clip_get_hidden_size(ctx);
        int mmproj_embd = clip_n_mmproj_embd(ctx);
        
        std::cout << "Encoding: n_tokens=" << n_tokens << ", hidden_dim=" << hidden_dim 
                  << ", mmproj_embd=" << mmproj_embd << std::endl;
        
        // Allocate embedding buffer based on actual token count
        // clip_embd_nbytes(ctx) uses default image size, but we need size for processed image
        size_t embd_count = (size_t)n_tokens * mmproj_embd;
        std::vector<float> embeddings(embd_count);
        
        // Encode image
        int64_t t0 = ggml_time_ms();
        if (!clip_image_encode(ctx, n_threads, img_f32, embeddings.data())) {
            clip_image_f32_batch_free(img_batch);
            return json{{"error", "Failed to encode image"}};
        }
        int64_t t1 = ggml_time_ms();
        
        std::cout << "Image encoded in " << (t1 - t0) << " ms" << std::endl;
        
        // Get position info for M-RoPE models (like Qwen3-VL)
        int nx_out = clip_n_output_tokens_x(ctx, img_f32);
        int ny_out = clip_n_output_tokens_y(ctx, img_f32);
        bool is_mrope = clip_is_mrope(ctx);
        
        clip_image_f32_batch_free(img_batch);
        
        // Build response
        json response = {
            {"embedding", embeddings},
            {"shape", {n_tokens, mmproj_embd}},
            {"n_tokens", n_tokens},
            {"hidden_dim", hidden_dim},
            {"mmproj_embd", mmproj_embd},
            {"image_size", {nx, ny}},
            {"encode_time_ms", t1 - t0}
        };
        
        // Add M-RoPE position info if applicable
        if (is_mrope) {
            response["mrope"] = {
                {"enabled", true},
                {"nx", nx_out},
                {"ny", ny_out}
            };
        }
        
        return response;
    }
};

static void print_usage(const char * prog) {
    std::cout << "Vision Encoder Server - Distributed VLM Inference\n\n"
              << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  -m, --model PATH     Path to vision encoder model (mmproj) [required]\n"
              << "  -p, --port PORT      Server port (default: 8081)\n"
              << "  -t, --threads N      Number of threads (default: 4)\n"
              << "  --host HOST          Host to bind (default: 0.0.0.0)\n"
              << "  --no-gpu             Disable GPU acceleration\n"
              << "  -h, --help           Show this help\n\n"
              << "Example:\n"
              << "  " << prog << " -m qwen3vl-mmproj.gguf -p 8081\n";
}

int main(int argc, char ** argv) {
    ggml_time_init();
    
    std::string model_path;
    std::string host = "0.0.0.0";
    int port = 8081;
    int n_threads = 4;
    bool use_gpu = true;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) model_path = argv[++i];
        } else if (arg == "-p" || arg == "--port") {
            if (i + 1 < argc) port = std::atoi(argv[++i]);
        } else if (arg == "-t" || arg == "--threads") {
            if (i + 1 < argc) n_threads = std::atoi(argv[++i]);
        } else if (arg == "--host") {
            if (i + 1 < argc) host = argv[++i];
        } else if (arg == "--no-gpu") {
            use_gpu = false;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    if (model_path.empty()) {
        std::cerr << "Error: Model path is required\n\n";
        print_usage(argv[0]);
        return 1;
    }
    
    // Initialize vision encoder
    vision_encoder_server encoder;
    encoder.n_threads = n_threads;
    
    if (!encoder.init(model_path.c_str(), use_gpu)) {
        std::cerr << "Failed to initialize vision encoder\n";
        return 1;
    }
    
    // Create HTTP server
    httplib::Server svr;
    
    // Health check endpoint
    svr.Get("/health", [&encoder](const httplib::Request&, httplib::Response& res) {
        json response = {
            {"status", "ok"},
            {"service", "vision-encoder-server"},
            {"model_loaded", encoder.ctx != nullptr}
        };
        res.set_content(response.dump(), "application/json");
    });
    
    // Model info endpoint
    svr.Get("/v1/model/info", [&encoder](const httplib::Request&, httplib::Response& res) {
        if (!encoder.ctx) {
            res.status = 503;
            res.set_content("{\"error\": \"Model not loaded\"}", "application/json");
            return;
        }
        
        json response = {
            {"hidden_size", clip_get_hidden_size(encoder.ctx)},
            {"image_size", clip_get_image_size(encoder.ctx)},
            {"patch_size", clip_get_patch_size(encoder.ctx)},
            {"mmproj_embd", clip_n_mmproj_embd(encoder.ctx)},
            {"is_mrope", clip_is_mrope(encoder.ctx)}
        };
        res.set_content(response.dump(), "application/json");
    });
    
    // Vision encode endpoint
    svr.Post("/v1/vision/encode", [&encoder](const httplib::Request& req, httplib::Response& res) {
        if (!encoder.ctx) {
            res.status = 503;
            res.set_content("{\"error\": \"Model not loaded\"}", "application/json");
            return;
        }
        
        try {
            auto body = json::parse(req.body);
            
            if (!body.contains("image")) {
                res.status = 400;
                json error = {{"error", "Missing 'image' field"}};
                res.set_content(error.dump(), "application/json");
                return;
            }
            
            std::string image_b64 = body["image"];
            
            // Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
            if (image_b64.find("data:") == 0) {
                auto pos = image_b64.find(",");
                if (pos != std::string::npos) {
                    image_b64 = image_b64.substr(pos + 1);
                }
            }
            
            auto result = encoder.encode_image_base64(image_b64);
            
            if (result.contains("error")) {
                res.status = 500;
            }
            
            res.set_content(result.dump(), "application/json");
            
        } catch (const json::parse_error& e) {
            res.status = 400;
            json error = {{"error", "Invalid JSON: " + std::string(e.what())}};
            res.set_content(error.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            json error = {{"error", e.what()}};
            res.set_content(error.dump(), "application/json");
        }
    });
    
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════╗\n";
    std::cout << "║         Vision Encoder Server v1.0                 ║\n";
    std::cout << "╠════════════════════════════════════════════════════╣\n";
    std::cout << "║  Listening on: " << host << ":" << port << std::string(35 - host.length() - std::to_string(port).length(), ' ') << "║\n";
    std::cout << "╠════════════════════════════════════════════════════╣\n";
    std::cout << "║  Endpoints:                                        ║\n";
    std::cout << "║    GET  /health              Health check          ║\n";
    std::cout << "║    GET  /v1/model/info       Model information     ║\n";
    std::cout << "║    POST /v1/vision/encode    Encode image          ║\n";
    std::cout << "╚════════════════════════════════════════════════════╝\n\n";
    
    if (!svr.listen(host.c_str(), port)) {
        std::cerr << "Failed to start server on " << host << ":" << port << std::endl;
        std::cerr << "Make sure the port is not already in use.\n";
        return 1;
    }
    
    return 0;
}
