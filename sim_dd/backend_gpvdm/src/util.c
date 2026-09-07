// util.c - numeric helpers.
//
// bernoulli_B/dB are ported from gpvdm libbasicmath/advmath.c (B(), dB()).
// The banded LU replaces gpvdm's UMFPACK sparse solve: the interleaved
// per-node variable ordering makes the Newton matrix banded, so a dense
// banded factorisation with partial pivoting is enough and keeps this
// program dependency-free.

#include "oled.h"

// gpvdm advmath.c: B(x) = x/(e^x - 1) with the small-x series
double bernoulli_B(double x)
{
	if (fabs(x) > 1e-40 && fabs(x) > 1e-8)
		return x / (exp(x) - 1.0);
	return 1.0 - x / 2.0 + x * x / 12.0 - pow(x, 4.0) / 720.0;
}

// gpvdm advmath.c: dB/dx
double bernoulli_dB(double x)
{
	if (fabs(x) > 1e-8)
	{
		double ex = exp(x);
		return 1.0 / (ex - 1.0) - x * ex / ((ex - 1.0) * (ex - 1.0));
	}
	return -1.0 / 2.0 + x / 6.0 - pow(x, 3.0) / 180.0 + pow(x, 5.0) / 5040.0;
}

void *xmalloc(size_t s)
{
	void *p = calloc(1, s);
	if (p == NULL)
	{
		fprintf(stderr, "out of memory\n");
		exit(1);
	}
	return p;
}

double **alloc2d(int a, int b)
{
	int i;
	double **p = xmalloc(sizeof(double *) * a);
	for (i = 0; i < a; i++)
		p[i] = xmalloc(sizeof(double) * b);
	return p;
}

// General banded LU with partial pivoting (LAPACK dgbtrf layout).
// AB has (2*kl+ku+1) rows of length n: AB[(kl+ku+i-j)*n + j] = A(i,j).
// Returns 0 on success, -1 on a singular pivot.
int band_lu_solve(double *AB, double *b, int n, int kl, int ku)
{
	int i, j, k;
	int kv = kl + ku;

	for (k = 0; k < n; k++)
	{
		int last = (k + kl < n - 1) ? k + kl : n - 1;
		int imax = k;
		double amax = fabs(AB[(size_t)(kv + k - k) * n + k]);
		for (i = k + 1; i <= last; i++)
		{
			double a = fabs(AB[(size_t)(kv + i - k) * n + k]);
			if (a > amax) { amax = a; imax = i; }
		}
		if (amax == 0.0)
			return -1;
		if (imax != k)
		{
			int jlast = (k + kv < n - 1) ? k + kv : n - 1;
			for (j = k; j <= jlast; j++)
			{
				double tmp = AB[(size_t)(kv + k - j) * n + j];
				AB[(size_t)(kv + k - j) * n + j] = AB[(size_t)(kv + imax - j) * n + j];
				AB[(size_t)(kv + imax - j) * n + j] = tmp;
			}
			double tmp = b[k]; b[k] = b[imax]; b[imax] = tmp;
		}
		for (i = k + 1; i <= last; i++)
		{
			double m = AB[(size_t)(kv + i - k) * n + k] / AB[(size_t)(kv + k - k) * n + k];
			AB[(size_t)(kv + i - k) * n + k] = 0.0;
			if (m != 0.0)
			{
				int jlast = (k + kv < n - 1) ? k + kv : n - 1;
				for (j = k + 1; j <= jlast; j++)
					AB[(size_t)(kv + i - j) * n + j] -= m * AB[(size_t)(kv + k - j) * n + j];
				b[i] -= m * b[k];
			}
		}
	}
	for (i = n - 1; i >= 0; i--)
	{
		double sum = b[i];
		int jlast = (i + kv < n - 1) ? i + kv : n - 1;
		for (j = i + 1; j <= jlast; j++)
			sum -= AB[(size_t)(kv + i - j) * n + j] * b[j];
		b[i] = sum / AB[(size_t)(kv + i - i) * n + i];
	}
	return 0;
}
