/* Writes an IVF VP9 stream whose size changes without a key frame, so later
 * frames predict from references of another size (scaled motion
 * compensation). Usage: resize <out.ivf> [bitdepth]  Built against libvpx. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vpx/vpx_encoder.h>
#include <vpx/vp8cx.h>

static void put32(FILE *f, unsigned v) { fputc(v, f); fputc(v >> 8, f); fputc(v >> 16, f); fputc(v >> 24, f); }
static void put16(FILE *f, unsigned v) { fputc(v, f); fputc(v >> 8, f); }

int main(int argc, char **argv) {
  if (argc < 2) return 1;
  FILE *out = fopen(argv[1], "wb");
  const int sizes[][2] = {{352, 288}, {264, 216}, {176, 144}, {352, 288}, {320, 240}, {400, 300}};
  const int n_sizes = 6, per = 6;
  vpx_codec_ctx_t ctx; vpx_codec_enc_cfg_t cfg;
  vpx_codec_enc_config_default(vpx_codec_vp9_cx(), &cfg, 0);
  cfg.g_w = 400; cfg.g_h = 300; cfg.g_timebase.num = 1; cfg.g_timebase.den = 30;
  cfg.rc_target_bitrate = 400; cfg.g_lag_in_frames = 0; cfg.kf_max_dist = 1000;
  cfg.g_w = sizes[0][0]; cfg.g_h = sizes[0][1];
  if (vpx_codec_enc_init(&ctx, vpx_codec_vp9_cx(), &cfg, 0)) { fprintf(stderr, "init\n"); return 1; }
  vpx_codec_control(&ctx, VP8E_SET_CPUUSED, 4);
  fwrite("DKIF", 1, 4, out); put16(out, 0); put16(out, 32); fwrite("VP90", 1, 4, out);
  put16(out, 400); put16(out, 300); put32(out, 30); put32(out, 1); put32(out, n_sizes * per); put32(out, 0);
  int frame = 0;
  for (int s = 0; s < n_sizes; s++) {
    int w = sizes[s][0], h = sizes[s][1];
    if (s) { cfg.g_w = w; cfg.g_h = h; if (vpx_codec_enc_config_set(&ctx, &cfg)) { fprintf(stderr, "set %d\n", s); return 1; } }
    vpx_image_t img; vpx_img_alloc(&img, VPX_IMG_FMT_I420, w, h, 16);
    for (int i = 0; i < per; i++, frame++) {
      for (int p = 0; p < 3; p++) {
        int pw = p ? (w + 1) / 2 : w, ph = p ? (h + 1) / 2 : h;
        for (int y = 0; y < ph; y++) for (int x = 0; x < pw; x++) {
          int X = x * (p ? 2 : 1) * 400 / w, Y = y * (p ? 2 : 1) * 300 / h;
          img.planes[p][y * img.stride[p] + x] = p ? (unsigned char)(128 + 40 * ((X / 40 + Y / 30 + frame) & 1) - p * 10)
            : (unsigned char)((X * 3 + Y * 2 + frame * 5) ^ ((X + frame * 3) * (Y + frame) >> 6));
        }
      }
      if (vpx_codec_encode(&ctx, &img, frame, 1, 0, VPX_DL_GOOD_QUALITY)) { fprintf(stderr, "enc\n"); return 1; }
      const vpx_codec_cx_pkt_t *pkt; vpx_codec_iter_t it = NULL;
      while ((pkt = vpx_codec_get_cx_data(&ctx, &it)))
        if (pkt->kind == VPX_CODEC_CX_FRAME_PKT) {
          put32(out, pkt->data.frame.sz); put32(out, pkt->data.frame.pts); put32(out, 0);
          fwrite(pkt->data.frame.buf, 1, pkt->data.frame.sz, out);
        }
    }
    vpx_img_free(&img);
  }
  vpx_codec_destroy(&ctx); fclose(out); return 0;
}
