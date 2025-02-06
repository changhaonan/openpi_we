# Convert the world engine's data into pi-0 format.


## Trouble-shot

The video encoder matters the result a lot. The default video encoder for lerobot is libsvtav1, which I installed fail on this machine. (May take more time to figure it out.) libx265 has a good compression ratio, but it may lose frames. libx264. works fine, but the video quality is bad and compression is bad.

PATH="$HOME/bin:$PATH" PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure \
  --prefix="$HOME/ffmpeg_build" \
  --pkg-config-flags="--static" \
  --extra-cflags="-I$HOME/ffmpeg_build/include" \
  --extra-ldflags="-L$HOME/ffmpeg_build/lib" \
  --extra-libs="-lpthread -lm" \
  --ld="g++" \
  --bindir="$HOME/bin" \
  --enable-gpl \
  --enable-libsvtav1 \
  --enable-libx264 \
  --enable-libx265 \
  --enable-nonfree && \
PATH="$HOME/bin:$PATH" make && \
make install && \
hash -r

In order to install ffmpeg with `libsvtav1`, we need to build ffmpeg from source. However, there is a recent change in SVT-AV1's API, which results in that if we still follows the old instruction. There will be a bug in compiling. So we need to install a old-version of SVT-AV1. (Checkout to branch r2.3.0)