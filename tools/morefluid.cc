#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

struct FluidHeader {
  float restParticlesPerMeter;
  int origNumParticles;
};

struct Particle {
  float p[3];
  float hv[3];
  float v[3];
};

int main(int argc, const char *argv[])
{
  if(argc != 3) {
    fprintf(stderr, "usage: %s [scale_factor] [num_buckets] < input > output\n",
	    argv[0]);
    exit(1);
  }

  int scale = atoi(argv[1]);
  int buckets = atoi(argv[2]);

  int scale3 = scale * scale * scale;

  fprintf(stderr, "running with scale=%d, buckets=%d\n", scale, buckets);

  FluidHeader h;
  size_t len;
  len = fread(&h, sizeof(FluidHeader), 1, stdin);
  assert(len == 1);

  fprintf(stderr, "read: rest=%f count=%d\n", h.restParticlesPerMeter, h.origNumParticles);

  h.restParticlesPerMeter *= scale;
  h.origNumParticles *= scale3;

  fwrite(&h, sizeof(FluidHeader), 1, stdout);

  // ranges hardcoded
  float dmin[3], dmax[3];
  dmin[0] = -0.065f;
  dmin[1] = -0.08f;
  dmin[2] = -0.065f;
  dmax[0] = 0.065f;
  dmax[1] = 0.1f;
  dmax[2] = 0.065f;

  // now read all the particles and shift/replicate them
  int max_particles = 1048576;
  Particle *p_in = (Particle *)malloc(max_particles * sizeof(Particle));
  Particle *p_out = (Particle *)malloc(max_particles * scale3 * sizeof(Particle));
  while((len = fread(p_in, sizeof(Particle), max_particles, stdin)) > 0) {
    fprintf(stderr, "read returned %zd particles\n", len);

    Particle *pp_in = p_in;
    Particle *pp_out = p_out;

    for(size_t i = 0; i < len; i++) {
      float rel[3];
      // map to [0,buckets)^3 cube
      for(int k = 0; k < 3; k++)
	rel[k] = buckets * (pp_in->p[k] - dmin[k]) / (dmax[k] - dmin[k]);
      //fprintf(stderr, "rel = (%8.4f, %8.4f, %8.4f)\n", rel[0], rel[1], rel[2]);
      
      // compact each bucket by scale
      for(int k = 0; k < 3; k++)
	rel[k] = truncf(rel[k]) + ((rel[k] - truncf(rel[k])) / scale);

      int dd[3];
      for(dd[0] = 0; dd[0] < scale; dd[0]++)
	for(dd[1] = 0; dd[1] < scale; dd[1]++)
	  for(dd[2] = 0; dd[2] < scale; dd[2]++) {
	    for(int k = 0; k < 3; k++) {
	      pp_out->p[k] = (dmin[k] + 
			      (((dmax[k] - dmin[k]) / buckets) * 
			       (rel[k] +
				(1.0 * dd[k] / scale))));
	      pp_out->hv[k] = pp_in->hv[k] / scale;
	      pp_out->v[k] = pp_in->v[k] / scale;
	    }
	    pp_out++;
	  }
      pp_in++;
    }

    size_t len2 = fwrite(p_out, sizeof(Particle), len * scale3, stdout);
    assert(len2 == (len * scale3));
  }
}
