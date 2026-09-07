// mesh.c - device discretisation.
//
// 1D mesh over the electrical layer stack (gpvdm's electrical mesh spans
// the device layers; contacts are ghost nodes beyond the ends).  Uniform
// spacing inside each layer, with per-layer point counts so thin transport
// layers can be refined independently.

#include "oled.h"

void mesh_build(struct device *dev, struct sim_config *cfg)
{
	int i, l, k;
	int N = 0;
	double x0 = 0.0;

	for (l = 0; l < dev->nlayers; l++)
	{
		if (dev->layers[l].points <= 0)
			dev->layers[l].points = cfg->points_per_layer;
		N += dev->layers[l].points;
	}
	dev->N = N;
	dev->x = xmalloc(sizeof(double) * N);
	dev->lay = xmalloc(sizeof(int) * N);

	i = 0;
	dev->L = 0.0;
	for (l = 0; l < dev->nlayers; l++)
	{
		int pts = dev->layers[l].points;
		double dy = dev->layers[l].thickness / pts;
		for (k = 0; k < pts; k++)
		{
			dev->x[i] = x0 + (k + 0.5) * dy;
			dev->lay[i] = l;
			i++;
		}
		x0 += dev->layers[l].thickness;
		dev->L += dev->layers[l].thickness;
	}

	// per-node material arrays
	dev->Eg = xmalloc(sizeof(double) * N);
	dev->Xi = xmalloc(sizeof(double) * N);
	dev->epsr = xmalloc(sizeof(double) * N);
	dev->Nad = xmalloc(sizeof(double) * N);
	for (i = 0; i < N; i++)
	{
		struct layer *la = &dev->layers[dev->lay[i]];
		dev->Eg[i] = la->Eg;
		dev->Xi[i] = la->Xi;
		dev->epsr[i] = la->epsr;
		dev->Nad[i] = la->Nad;
	}

	// solver state + derived arrays
	dev->phi = xmalloc(sizeof(double) * N);
	dev->xn = xmalloc(sizeof(double) * N);
	dev->xp = xmalloc(sizeof(double) * N);
	dev->n = xmalloc(sizeof(double) * N);
	dev->p = xmalloc(sizeof(double) * N);
	dev->dn = xmalloc(sizeof(double) * N);
	dev->dp = xmalloc(sizeof(double) * N);
	dev->wn = xmalloc(sizeof(double) * N);
	dev->wp = xmalloc(sizeof(double) * N);
	dev->mun = xmalloc(sizeof(double) * N);
	dev->mup = xmalloc(sizeof(double) * N);
	dev->Rfree = xmalloc(sizeof(double) * N);
	dev->Rsrh = xmalloc(sizeof(double) * N);
	dev->Rauger = xmalloc(sizeof(double) * N);
	dev->Rtot = xmalloc(sizeof(double) * N);
	dev->nt_all = xmalloc(sizeof(double) * N);
	dev->pt_all = xmalloc(sizeof(double) * N);
	dev->Jn = xmalloc(sizeof(double) * N);
	dev->Jp = xmalloc(sizeof(double) * N);
	dev->n_eq = xmalloc(sizeof(double) * N);
	dev->p_eq = xmalloc(sizeof(double) * N);

	// trap bands: global band count = max over layers (bands are inert in
	// layers with fewer/no bands)
	dev->nbands = 0;
	for (l = 0; l < dev->nlayers; l++)
	{
		if (dev->layers[l].dosn.nbands > dev->nbands) dev->nbands = dev->layers[l].dosn.nbands;
		if (dev->layers[l].dosp.nbands > dev->nbands) dev->nbands = dev->layers[l].dosp.nbands;
	}
	if (dev->nbands > 0)
	{
		dev->xt = alloc2d(dev->nbands, N);
		dev->xpt = alloc2d(dev->nbands, N);
		dev->nt = alloc2d(dev->nbands, N);
		dev->dnt = alloc2d(dev->nbands, N);
		dev->pt = alloc2d(dev->nbands, N);
		dev->dpt = alloc2d(dev->nbands, N);
		dev->srh_n_r1 = alloc2d(dev->nbands, N);
		dev->srh_n_r2 = alloc2d(dev->nbands, N);
		dev->srh_n_r3 = alloc2d(dev->nbands, N);
		dev->srh_n_r4 = alloc2d(dev->nbands, N);
		dev->srh_p_r1 = alloc2d(dev->nbands, N);
		dev->srh_p_r2 = alloc2d(dev->nbands, N);
		dev->srh_p_r3 = alloc2d(dev->nbands, N);
		dev->srh_p_r4 = alloc2d(dev->nbands, N);
		dev->dsrh_n_r1 = alloc2d(dev->nbands, N);
		dev->dsrh_n_r2 = alloc2d(dev->nbands, N);
		dev->dsrh_n_r3 = alloc2d(dev->nbands, N);
		dev->dsrh_n_r4 = alloc2d(dev->nbands, N);
		dev->dsrh_p_r1 = alloc2d(dev->nbands, N);
		dev->dsrh_p_r2 = alloc2d(dev->nbands, N);
		dev->dsrh_p_r3 = alloc2d(dev->nbands, N);
		dev->dsrh_p_r4 = alloc2d(dev->nbands, N);
	}
}
