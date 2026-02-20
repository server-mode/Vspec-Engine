#ifndef VSPEC_VALIDATION_PERPLEXITY_H
#define VSPEC_VALIDATION_PERPLEXITY_H

#include <stddef.h>

float vspec_perplexity_from_nll(const float* nll, size_t count);
float vspec_perplexity_from_logits(const float* logits, size_t vocab, size_t count);

#endif
