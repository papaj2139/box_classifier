#include <errno.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <time.h>
#include <unistd.h>

#define OUTPUT_DEFAULT "dataset/wild"
#define FINAL_SIZE 32
#define MAX_CANVAS 160

static const int SIZES[] = {48, 64, 80, 96, 112, 128, 160};
static const uint8_t PAPER_LEVELS[] = {255, 252, 231, 229, 233, 245, 247, 250, 243, 244, 242, 220, 211};
static const uint8_t FILL_LEVELS[] = {0, 26, 51, 77, 85, 102};

typedef struct {
    uint64_t state;
} Rng;

typedef struct {
    int w;
    int h;
    uint8_t *pix;
} GrayImage;

typedef struct {
    const char *out_dir;
    size_t empty_count;
    size_t filled_count;
    uint64_t seed;
    int threads;
} Options;

static uint64_t splitmix64_next(uint64_t *x) {
    uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static void rng_seed(Rng *r, uint64_t seed) {
    if (!seed) seed = 0x123456789abcdefULL;
    uint64_t x = seed;
    r->state = splitmix64_next(&x);
    if (!r->state) r->state = 0xdeadbeefcafef00dULL;
}

static uint32_t rng_u32(Rng *r) {
    uint64_t x = r->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    r->state = x;
    return (uint32_t)((x * 2685821657736338717ULL) >> 32);
}

static float rng_f32(Rng *r) {
    return (float)(rng_u32(r) & 0x00ffffffu) * (1.0f / 16777216.0f);
}

static int rng_int(Rng *r, int lo, int hi) {
    if (hi <= lo) return lo;
    uint32_t span = (uint32_t)(hi - lo + 1);
    return lo + (int)(rng_u32(r) % span);
}

static float rng_range(Rng *r, float lo, float hi) {
    return lo + (hi - lo) * rng_f32(r);
}

static uint8_t clamp_u8_int(int v) {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return (uint8_t)v;
}

static int clamp_i32(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static size_t image_area(int w, int h) {
    return (size_t)w * (size_t)h;
}

static void img_fill(GrayImage *img, uint8_t value) {
    memset(img->pix, value, image_area(img->w, img->h));
}

static inline uint8_t img_get(const GrayImage *img, int x, int y) {
    return img->pix[(size_t)y * (size_t)img->w + (size_t)x];
}

static inline void img_set(const GrayImage *img, int x, int y, uint8_t value) {
    if (x < 0 || y < 0 || x >= img->w || y >= img->h) return;
    img->pix[(size_t)y * (size_t)img->w + (size_t)x] = value;
}

static inline void img_blend(const GrayImage *img, int x, int y, uint8_t value, int alpha) {
    if (x < 0 || y < 0 || x >= img->w || y >= img->h) return;
    size_t idx = (size_t)y * (size_t)img->w + (size_t)x;
    int cur = img->pix[idx];
    int out = (cur * (255 - alpha) + value * alpha) / 255;
    img->pix[idx] = clamp_u8_int(out);
}

static void img_filled_rect(GrayImage *img, int x1, int y1, int x2, int y2, uint8_t value) {
    x1 = clamp_i32(x1, 0, img->w - 1);
    y1 = clamp_i32(y1, 0, img->h - 1);
    x2 = clamp_i32(x2, 0, img->w - 1);
    y2 = clamp_i32(y2, 0, img->h - 1);
    if (x2 < x1 || y2 < y1) return;
    for (int y = y1; y <= y2; y++) {
        size_t row = (size_t)y * (size_t)img->w;
        for (int x = x1; x <= x2; x++) {
            img->pix[row + (size_t)x] = value;
        }
    }
}

static void img_draw_circle(GrayImage *img, int cx, int cy, int rx, int ry, uint8_t value) {
    if (rx <= 0 || ry <= 0) return;
    int x1 = clamp_i32(cx - rx, 0, img->w - 1);
    int y1 = clamp_i32(cy - ry, 0, img->h - 1);
    int x2 = clamp_i32(cx + rx, 0, img->w - 1);
    int y2 = clamp_i32(cy + ry, 0, img->h - 1);
    float inv_rx2 = 1.0f / ((float)rx * (float)rx);
    float inv_ry2 = 1.0f / ((float)ry * (float)ry);
    for (int y = y1; y <= y2; y++) {
        for (int x = x1; x <= x2; x++) {
            float dx = (float)(x - cx);
            float dy = (float)(y - cy);
            if (dx * dx * inv_rx2 + dy * dy * inv_ry2 <= 1.0f) {
                img_set(img, x, y, value);
            }
        }
    }
}

static void img_blend_circle(GrayImage *img, int cx, int cy, int rx, int ry, uint8_t value, int alpha) {
    if (rx <= 0 || ry <= 0) return;
    int x1 = clamp_i32(cx - rx, 0, img->w - 1);
    int y1 = clamp_i32(cy - ry, 0, img->h - 1);
    int x2 = clamp_i32(cx + rx, 0, img->w - 1);
    int y2 = clamp_i32(cy + ry, 0, img->h - 1);
    float inv_rx2 = 1.0f / ((float)rx * (float)rx);
    float inv_ry2 = 1.0f / ((float)ry * (float)ry);
    for (int y = y1; y <= y2; y++) {
        for (int x = x1; x <= x2; x++) {
            float dx = (float)(x - cx);
            float dy = (float)(y - cy);
            float v = dx * dx * inv_rx2 + dy * dy * inv_ry2;
            if (v <= 1.0f) {
                int local_alpha = (int)(alpha * (1.0f - v));
                if (local_alpha > 0) img_blend(img, x, y, value, local_alpha);
            }
        }
    }
}

static void img_draw_line(GrayImage *img, int x0, int y0, int x1, int y1, uint8_t value, int width) {
    int dx = abs(x1 - x0);
    int sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0);
    int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    int radius = width / 2;

    for (;;) {
        for (int oy = -radius; oy <= radius; oy++) {
            for (int ox = -radius; ox <= radius; ox++) {
                img_set(img, x0 + ox, y0 + oy, value);
            }
        }
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y0 += sy;
        }
    }
}

static void draw_box(GrayImage *img, int x, int y, int w, int h, int line_width, Rng *rng) {
    int jitter = 1;
    int x0 = x + rng_int(rng, -jitter, jitter);
    int y0 = y + rng_int(rng, -jitter, jitter);
    int x1 = x + w + rng_int(rng, -jitter, jitter);
    int y1 = y + rng_int(rng, -jitter, jitter);
    int x2 = x + w + rng_int(rng, -jitter, jitter);
    int y2 = y + h + rng_int(rng, -jitter, jitter);
    int x3 = x + rng_int(rng, -jitter, jitter);
    int y3 = y + h + rng_int(rng, -jitter, jitter);
    int x4 = x + rng_int(rng, -jitter, jitter);
    int y4 = y + rng_int(rng, -jitter, jitter);
    img_draw_line(img, x0, y0, x1, y1, 0, line_width);
    img_draw_line(img, x1, y1, x2, y2, 0, line_width);
    img_draw_line(img, x2, y2, x3, y3, 0, line_width);
    img_draw_line(img, x3, y3, x4, y4, 0, line_width);
}

static int max_int(int a, int b) { return a > b ? a : b; }
static int min_int(int a, int b) { return a < b ? a : b; }

static void fill_box(GrayImage *img, int x, int y, int w, int h, int style, uint8_t fill_value, Rng *rng) {
    int margin = max_int(2, min_int(w, h) / 8);
    int x1 = x + margin;
    int y1 = y + margin;
    int x2 = x + w - margin;
    int y2 = y + h - margin;
    x1 = clamp_i32(x1, 0, img->w - 1);
    y1 = clamp_i32(y1, 0, img->h - 1);
    x2 = clamp_i32(x2, 0, img->w - 1);
    y2 = clamp_i32(y2, 0, img->h - 1);
    if (x2 < x1 || y2 < y1) return;

    switch (style) {
        case 0: // solid
            img_filled_rect(img, x1, y1, x2, y2, fill_value);
            break;
        case 1: // x
        {
            int lw = max_int(2, min_int(w, h) / 10);
            img_draw_line(img, x1, y1, x2, y2, fill_value, lw);
            img_draw_line(img, x2, y1, x1, y2, fill_value, lw);
            break;
        }
        case 2: // check
        {
            int lw = max_int(2, min_int(w, h) / 8);
            img_draw_line(img, x + w / 5, y + h / 2, x + w / 3, y + (h * 4) / 5, fill_value, lw);
            img_draw_line(img, x + w / 3, y + (h * 4) / 5, x + (w * 17) / 20, y + h / 5, fill_value, lw);
            break;
        }
        case 3: // scribble
        {
            int cx = x1;
            int cy = y1;
            int iters = rng_int(rng, 10, 25);
            for (int i = 0; i < iters; i++) {
                int nx = rng_int(rng, x1, x2);
                int ny = rng_int(rng, y1, y2);
                img_draw_line(img, cx, cy, nx, ny, fill_value, rng_int(rng, 1, 3));
                cx = nx;
                cy = ny;
            }
            break;
        }
        case 4: // diagonal
        {
            int lw = max_int(1, min_int(w, h) / 12);
            int step = max_int(3, min_int(w, h) / 6);
            int bound = max_int(w, h);
            for (int i = -bound; i < bound * 2; i += step) {
                img_draw_line(img, x1 + i, y1, x1 + i - h, y2, fill_value, lw);
            }
            break;
        }
        case 5: // circle
            img_draw_circle(img, x + w / 2, y + h / 2, max_int(1, w / 2 - margin), max_int(1, h / 2 - margin), fill_value);
            break;
        case 6: // ink spread
        {
            int cx = x + w / 2;
            int cy = y + h / 2;
            int blobs = rng_int(rng, 3, 7);
            for (int i = 0; i < blobs; i++) {
                int r = rng_int(rng, min_int(w, h) / 4, max_int(2, min_int(w, h) / 2));
                img_draw_circle(img, cx + rng_int(rng, -2, 2), cy + rng_int(rng, -2, 2), r, r, fill_value);
            }
            break;
        }
    }
}

static void add_noise(GrayImage *img, Rng *rng, float intensity) {
    int amplitude = (int)(intensity * 255.0f);
    if (amplitude < 1) amplitude = 1;
    size_t total = image_area(img->w, img->h);
    for (size_t i = 0; i < total; i++) {
        int delta = rng_int(rng, -amplitude, amplitude);
        img->pix[i] = clamp_u8_int((int)img->pix[i] + delta);
    }
}

static void box_blur(const GrayImage *src, GrayImage *dst) {
    for (int y = 0; y < src->h; y++) {
        for (int x = 0; x < src->w; x++) {
            int sum = 0;
            int count = 0;
            for (int oy = -1; oy <= 1; oy++) {
                int sy = y + oy;
                if (sy < 0 || sy >= src->h) continue;
                for (int ox = -1; ox <= 1; ox++) {
                    int sx = x + ox;
                    if (sx < 0 || sx >= src->w) continue;
                    sum += img_get(src, sx, sy);
                    count++;
                }
            }
            dst->pix[(size_t)y * (size_t)src->w + (size_t)x] = (uint8_t)(sum / count);
        }
    }
}

static void sharpen(const GrayImage *src, GrayImage *dst) {
    for (int y = 0; y < src->h; y++) {
        for (int x = 0; x < src->w; x++) {
            int c = img_get(src, x, y);
            int up = img_get(src, x, y > 0 ? y - 1 : y);
            int down = img_get(src, x, y + 1 < src->h ? y + 1 : y);
            int left = img_get(src, x > 0 ? x - 1 : x, y);
            int right = img_get(src, x + 1 < src->w ? x + 1 : x, y);
            int out = 5 * c - up - down - left - right;
            dst->pix[(size_t)y * (size_t)src->w + (size_t)x] = clamp_u8_int(out);
        }
    }
}

static void rotate_image(const GrayImage *src, GrayImage *dst, float radians, uint8_t bg) {
    float cs = cosf(radians);
    float sn = sinf(radians);
    float cx = (float)(src->w - 1) * 0.5f;
    float cy = (float)(src->h - 1) * 0.5f;

    for (int y = 0; y < src->h; y++) {
        for (int x = 0; x < src->w; x++) {
            float fx = (float)x - cx;
            float fy = (float)y - cy;
            float sx = cs * fx + sn * fy + cx;
            float sy = -sn * fx + cs * fy + cy;

            int ix = (int)floorf(sx);
            int iy = (int)floorf(sy);
            float tx = sx - (float)ix;
            float ty = sy - (float)iy;

            uint8_t value = bg;
            if (ix >= 0 && iy >= 0 && ix + 1 < src->w && iy + 1 < src->h) {
                int p00 = img_get(src, ix, iy);
                int p10 = img_get(src, ix + 1, iy);
                int p01 = img_get(src, ix, iy + 1);
                int p11 = img_get(src, ix + 1, iy + 1);
                float a = (1.0f - tx) * (1.0f - ty) * (float)p00;
                float b = tx * (1.0f - ty) * (float)p10;
                float c = (1.0f - tx) * ty * (float)p01;
                float d = tx * ty * (float)p11;
                value = clamp_u8_int((int)(a + b + c + d + 0.5f));
            }
            dst->pix[(size_t)y * (size_t)src->w + (size_t)x] = value;
        }
    }
}

static void resize_bilinear(const GrayImage *src, GrayImage *dst) {
    float scale_x = (float)(src->w - 1) / (float)(dst->w - 1);
    float scale_y = (float)(src->h - 1) / (float)(dst->h - 1);
    for (int y = 0; y < dst->h; y++) {
        float sy = scale_y * (float)y;
        int y0 = (int)floorf(sy);
        int y1 = y0 + 1;
        float ty = sy - (float)y0;
        if (y1 >= src->h) y1 = src->h - 1;
        for (int x = 0; x < dst->w; x++) {
            float sx = scale_x * (float)x;
            int x0 = (int)floorf(sx);
            int x1 = x0 + 1;
            float tx = sx - (float)x0;
            if (x1 >= src->w) x1 = src->w - 1;
            int p00 = img_get(src, x0, y0);
            int p10 = img_get(src, x1, y0);
            int p01 = img_get(src, x0, y1);
            int p11 = img_get(src, x1, y1);
            float a = (1.0f - tx) * (1.0f - ty) * (float)p00;
            float b = tx * (1.0f - ty) * (float)p10;
            float c = (1.0f - tx) * ty * (float)p01;
            float d = tx * ty * (float)p11;
            dst->pix[(size_t)y * (size_t)dst->w + (size_t)x] = clamp_u8_int((int)(a + b + c + d + 0.5f));
        }
    }
}

static void apply_background_texture(GrayImage *img, Rng *rng) {
    if (rng_f32(rng) >= 0.15f) return;

    uint8_t smudge = rng_f32(rng) < 0.5f ? 188 : 172;
    int cx = rng_int(rng, 0, img->w - 1);
    int cy = rng_int(rng, 0, img->h - 1);
    int r = rng_int(rng, img->w / 4, img->w);
    img_blend_circle(img, cx, cy, r, r, smudge, 40);
}

static void apply_grid(GrayImage *img, Rng *rng) {
    if (rng_f32(rng) >= 0.15f) return;
    int spacing = rng_int(rng, 10, 30);
    uint8_t color = (uint8_t)(rng_f32(rng) < 0.33f ? 204 : (rng_f32(rng) < 0.5f ? 221 : 187));
    for (int i = 0; i < img->w; i += spacing) {
        img_draw_line(img, i, 0, i, img->h - 1, color, 1);
        img_draw_line(img, 0, i, img->w - 1, i, color, 1);
    }
}

static void apply_random_marks(GrayImage *img, Rng *rng) {
    int marks = rng_int(rng, 1, 4);
    for (int i = 0; i < marks; i++) {
        if (rng_f32(rng) < 0.4f) {
            int x = rng_int(rng, 0, img->w - 1);
            int y = rng_int(rng, 0, img->h - 1);
            img_draw_line(img, x, y,
                          x + rng_int(rng, -20, 20),
                          y + rng_int(rng, -20, 20),
                          0, 1);
        }
    }
}

static uint64_t sample_seed(uint64_t base_seed, size_t index, int label) {
    uint64_t x = base_seed ^ 0x9e3779b97f4a7c15ULL;
    x ^= (uint64_t)index * 0xbf58476d1ce4e5b9ULL;
    x ^= (uint64_t)(label + 1) * 0x94d049bb133111ebULL;
    return splitmix64_next(&x);
}

static int random_size(Rng *rng) {
    size_t n = sizeof(SIZES) / sizeof(SIZES[0]);
    return SIZES[rng_int(rng, 0, (int)n - 1)];
}

static uint8_t random_paper(Rng *rng) {
    size_t n = sizeof(PAPER_LEVELS) / sizeof(PAPER_LEVELS[0]);
    return PAPER_LEVELS[rng_int(rng, 0, (int)n - 1)];
}

static uint8_t random_fill(Rng *rng) {
    size_t n = sizeof(FILL_LEVELS) / sizeof(FILL_LEVELS[0]);
    return FILL_LEVELS[rng_int(rng, 0, (int)n - 1)];
}

static int random_style(Rng *rng) {
    return rng_int(rng, 0, 6);
}

static void generate_sample(GrayImage *canvas, GrayImage *scratch, uint8_t *final32, Rng *rng, int label) {
    int size = random_size(rng);
    canvas->w = size;
    canvas->h = size;
    scratch->w = size;
    scratch->h = size;

    img_fill(canvas, random_paper(rng));
    apply_background_texture(canvas, rng);

    if (rng_f32(rng) < 0.15f) {
        apply_grid(canvas, rng);
    }

    float box_ratio = rng_range(rng, 0.3f, 0.8f);
    int box_w = (int)(size * box_ratio);
    int box_h = (int)(box_w * rng_range(rng, 0.8f, 1.2f));
    box_w = clamp_i32(box_w, 8, size - 4);
    box_h = clamp_i32(box_h, 8, size - 4);
    int box_x = rng_int(rng, 2, max_int(2, size - box_w - 4));
    int box_y = rng_int(rng, 2, max_int(2, size - box_h - 4));

    draw_box(canvas, box_x, box_y, box_w, box_h, rng_int(rng, 1, 3), rng);
    if (label == 1) {
        fill_box(canvas, box_x, box_y, box_w, box_h, random_style(rng), random_fill(rng), rng);
    }

    if (rng_f32(rng) < 0.2f) {
        float angle = rng_range(rng, -7.0f, 7.0f) * (float)M_PI / 180.0f;
        rotate_image(canvas, scratch, angle, random_paper(rng));
        uint8_t *swap = canvas->pix;
        canvas->pix = scratch->pix;
        scratch->pix = swap;
    }

    apply_random_marks(canvas, rng);

    if (rng_f32(rng) < 0.3f) {
        box_blur(canvas, scratch);
        uint8_t *swap = canvas->pix;
        canvas->pix = scratch->pix;
        scratch->pix = swap;
    }
    if (rng_f32(rng) < 0.2f) {
        sharpen(canvas, scratch);
        uint8_t *swap = canvas->pix;
        canvas->pix = scratch->pix;
        scratch->pix = swap;
    }

    add_noise(canvas, rng, rng_range(rng, 0.02f, 0.08f));

    GrayImage src = {canvas->w, canvas->h, canvas->pix};
    GrayImage dst = {FINAL_SIZE, FINAL_SIZE, final32};
    resize_bilinear(&src, &dst);
}

static int ensure_dir(const char *path) {
    if (mkdir(path, 0777) == 0) return 0;
    if (errno == EEXIST) return 0;
    return -1;
}

static int mkdir_p(const char *path) {
    char tmp[PATH_MAX];
    size_t len = strlen(path);
    if (len >= sizeof(tmp)) return -1;
    strcpy(tmp, path);

    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (ensure_dir(tmp) != 0) return -1;
            *p = '/';
        }
    }
    return ensure_dir(tmp);
}

static int remove_tree(const char *path) {
    DIR *dir = opendir(path);
    if (!dir) {
        if (errno == ENOENT) return 0;
        return -1;
    }

    struct dirent *entry;
    char buf[PATH_MAX];
    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;
        if (snprintf(buf, sizeof(buf), "%s/%s", path, entry->d_name) >= (int)sizeof(buf)) {
            closedir(dir);
            return -1;
        }

        struct stat st;
        if (stat(buf, &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) {
            if (remove_tree(buf) != 0) {
                closedir(dir);
                return -1;
            }
        } else {
            unlink(buf);
        }
    }
    closedir(dir);
    return rmdir(path);
}

static int save_pgm(const char *path, const uint8_t *pix, int w, int h) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    if (fprintf(f, "P5\n%d %d\n255\n", w, h) < 0) {
        fclose(f);
        return -1;
    }
    size_t total = (size_t)w * (size_t)h;
    if (fwrite(pix, 1, total, f) != total) {
        fclose(f);
        return -1;
    }
    fclose(f);
    return 0;
}

static void print_usage(const char *argv0) {
    printf("Usage: %s [--out DIR] [--empty N] [--filled N] [--seed N] [--threads N]\n", argv0);
    printf("Defaults: --out %s --empty 10000 --filled 10000\n", OUTPUT_DEFAULT);
}

int main(int argc, char **argv) {
    Options opt = {
        .out_dir = OUTPUT_DEFAULT,
        .empty_count = 10000,
        .filled_count = 10000,
        .seed = (uint64_t)time(NULL),
        .threads = 0,
    };

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            opt.out_dir = argv[++i];
        } else if (strcmp(argv[i], "--empty") == 0 && i + 1 < argc) {
            opt.empty_count = (size_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--filled") == 0 && i + 1 < argc) {
            opt.filled_count = (size_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            opt.seed = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            opt.threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (opt.empty_count == 0 && opt.filled_count == 0) {
        fprintf(stderr, "Nothing to generate.\n");
        return 1;
    }

    if (remove_tree(opt.out_dir) != 0 && errno != ENOENT) {
        fprintf(stderr, "Failed to clear output dir: %s\n", opt.out_dir);
        return 1;
    }
    if (mkdir_p(opt.out_dir) != 0) {
        fprintf(stderr, "Failed to create output dir: %s\n", opt.out_dir);
        return 1;
    }

    char empty_dir[PATH_MAX];
    char filled_dir[PATH_MAX];
    if (snprintf(empty_dir, sizeof(empty_dir), "%s/empty", opt.out_dir) >= (int)sizeof(empty_dir) ||
        snprintf(filled_dir, sizeof(filled_dir), "%s/filled", opt.out_dir) >= (int)sizeof(filled_dir)) {
        fprintf(stderr, "Output path too long.\n");
        return 1;
    }
    if (mkdir_p(empty_dir) != 0 || mkdir_p(filled_dir) != 0) {
        fprintf(stderr, "Failed to create class directories.\n");
        return 1;
    }

    if (opt.threads > 0) omp_set_num_threads(opt.threads);
    size_t total = opt.empty_count + opt.filled_count;
    size_t progress = 0;
    size_t progress_step = total >= 10000 ? 10000 : (total / 10 ? total / 10 : 1);

    printf("Generating %zu empty + %zu filled samples into %s\n",
           opt.empty_count, opt.filled_count, opt.out_dir);

    #pragma omp parallel
    {
        uint8_t *canvas_buf = (uint8_t *)malloc(MAX_CANVAS * MAX_CANVAS);
        uint8_t *scratch_buf = (uint8_t *)malloc(MAX_CANVAS * MAX_CANVAS);
        uint8_t final_buf[FINAL_SIZE * FINAL_SIZE];

        GrayImage canvas = {0, 0, canvas_buf};
        GrayImage scratch = {0, 0, scratch_buf};

        #pragma omp for schedule(dynamic, 1)
        for (size_t i = 0; i < total; i++) {
            int label = (i < opt.empty_count) ? 0 : 1;
            size_t local_idx = (label == 0) ? i : (i - opt.empty_count);
            uint64_t seed = sample_seed(opt.seed, i, label);
            Rng rng;
            rng_seed(&rng, seed);

            generate_sample(&canvas, &scratch, final_buf, &rng, label);

            char path[PATH_MAX];
            int n = snprintf(path, sizeof(path), "%s/%s/%s_%08zu.pgm",
                             opt.out_dir,
                             label ? "filled" : "empty",
                             label ? "filled" : "empty",
                             local_idx);
            if (n > 0 && n < (int)sizeof(path)) {
                save_pgm(path, final_buf, FINAL_SIZE, FINAL_SIZE);
            }

            size_t done;
            #pragma omp atomic capture
            done = ++progress;
            if (done % progress_step == 0 || done == total) {
                #pragma omp critical
                {
                    printf("  Progress: %zu/%zu\n", done, total);
                }
            }
        }

        free(canvas_buf);
        free(scratch_buf);
    }

    char labels_path[PATH_MAX];
    if (snprintf(labels_path, sizeof(labels_path), "%s/labels.txt", opt.out_dir) >= (int)sizeof(labels_path)) {
        fprintf(stderr, "labels path too long\n");
        return 1;
    }
    FILE *labels = fopen(labels_path, "w");
    if (!labels) {
        fprintf(stderr, "Failed to write %s\n", labels_path);
        return 1;
    }
    for (size_t i = 0; i < opt.empty_count; i++) {
        fprintf(labels, "empty/empty_%08zu.pgm 0\n", i);
    }
    for (size_t i = 0; i < opt.filled_count; i++) {
        fprintf(labels, "filled/filled_%08zu.pgm 1\n", i);
    }
    fclose(labels);

    printf("Done. Wrote %zu samples to %s\n", total, opt.out_dir);
    return 0;
}
