# Index
- [[#Structural Similarity Index (SSIM)]]
- [[#Fréchet inception distance]]
- [[#Kernel Inception Distance]]
- [[#Human perceptual]]



# Structural Similarity Index (SSIM)
Wikipedia: https://en.wikipedia.org/wiki/Structural_similarity

Pytorch documentation: https://pytorch.org/ignite/generated/ignite.metrics.SSIM.html

Coding example: https://github.com/pranjaldatta/SSIM-PyTorch/blob/master/SSIM_notebook.ipynb 

Original paper: Wang, Zhou; Bovik, A.C.; Sheikh, H.R.; Simoncelli, E.P. (2004-04-01). "Image quality assessment: from error visibility to structural similarity". IEEE Transactions on Image Processing. 13 (4): 600–612.




# Fréchet inception distance
Wikipedia: https://en.wikipedia.org/wiki/Fréchet_inception_distance

Tutorial (numpy): https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/

Tutorial (tensorflow): https://wandb.ai/ayush-thakur/gan-evaluation/reports/How-to-Evaluate-GANs-using-Frechet-Inception-Distance-FID---Vmlldzo0MTAxOTI

Pytorch official documentation: https://pytorch.org/ignite/generated/ignite.metrics.FID.html

Useful repository (maybe): https://github.com/taki0112/GAN_Metrics-Tensorflow

Original paper: Heusel, Martin; Ramsauer, Hubert; Unterthiner, Thomas; Nessler, Bernhard; Hochreiter, Sepp (2017). "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium". Advances in Neural Information Processing Systems. 


# Kernel Inception Distance
???


# Human perceptual
A human receive two images:
- generated images (syntethic)
- real image (ground-truth)
She has to:
- choose which is the syntethic image
- rate the similarity with a grade from 0 (no similarity at all) to 10 (no differences).
