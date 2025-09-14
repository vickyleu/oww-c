#include "oww.h"
#include <alsa/asoundlib.h>
#include <vector>
#include <cstdio>
static void die(const char* m,int e=0){ if(e) fprintf(stderr,"%s: %s\n",m,snd_strerror(e)); else fprintf(stderr,"%s\n",m); exit(1); }

int main(int argc,char**argv){
  if(argc<5){ fprintf(stderr,"usage: %s <melspec.onnx> <embedding_model.onnx> <detector.onnx> <device>\n",argv[0]); return 2; }
  oww_handle* h= oww_create(argv[1],argv[2],argv[3],1,0.5f);
  snd_pcm_t* pcm=nullptr; int err=snd_pcm_open(&pcm, argv[4], SND_PCM_STREAM_CAPTURE, 0);
  if(err<0) die("open",err);
  err=snd_pcm_set_params(pcm,SND_PCM_FORMAT_S16_LE,SND_PCM_ACCESS_RW_INTERLEAVED,1,16000,1,200000);
  if(err<0) die("set_params",err);
  size_t chunk=oww_recommended_chunk();
  std::vector<short> buf(chunk);
  while(true){
    snd_pcm_sframes_t n=snd_pcm_readi(pcm,buf.data(),chunk);
    if(n<0){ snd_pcm_recover(pcm,(int)n,1); continue; }
    if((size_t)n<chunk) continue;
    if(oww_process_i16(h, buf.data(), chunk)){
      printf("WAKE (score=%.3f)\n", oww_last_score(h)); fflush(stdout);
    }
  }
  snd_pcm_close(pcm); oww_destroy(h); return 0;
}
