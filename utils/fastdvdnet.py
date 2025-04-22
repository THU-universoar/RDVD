"""
Different utilities such as orthogonalization of weights, initialization of
loggers, etc
"""
import os
import subprocess
import glob
import logging
from random import choices # requires Python >= 3.6
import numpy as np
import cv2
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

"""
Different common functions for training the models.
"""

import time
import torchvision.utils as tutils


IMAGETYPES = ('*.bmp', '*.png', '*.jpg', '*.jpeg', '*.tif') # Supported image types


def normalize_augment(datain, ctrl_fr_idx):
	'''Normalizes and augments an input patch of dim [N, num_frames, C. H, W] in [0., 255.] to \
		[N, num_frames*C. H, W] in  [0., 1.]. It also returns the central frame of the temporal \
		patch as a ground truth.
	'''
	def transform(sample):
		# define transformations
		do_nothing = lambda x: x
		do_nothing.__name__ = 'do_nothing'
		flipud = lambda x: torch.flip(x, dims=[2])
		flipud.__name__ = 'flipup'
		rot90 = lambda x: torch.rot90(x, k=1, dims=[2, 3])
		rot90.__name__ = 'rot90'
		rot90_flipud = lambda x: torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[2])
		rot90_flipud.__name__ = 'rot90_flipud'
		rot180 = lambda x: torch.rot90(x, k=2, dims=[2, 3])
		rot180.__name__ = 'rot180'
		rot180_flipud = lambda x: torch.flip(torch.rot90(x, k=2, dims=[2, 3]), dims=[2])
		rot180_flipud.__name__ = 'rot180_flipud'
		rot270 = lambda x: torch.rot90(x, k=3, dims=[2, 3])
		rot270.__name__ = 'rot270'
		rot270_flipud = lambda x: torch.flip(torch.rot90(x, k=3, dims=[2, 3]), dims=[2])
		rot270_flipud.__name__ = 'rot270_flipud'
		add_csnt = lambda x: x + torch.normal(mean=torch.zeros(x.size()[0], 1, 1, 1), \
								 std=(5/255.)).expand_as(x).to(x.device)
		add_csnt.__name__ = 'add_csnt'

		# define transformations and their frequency, then pick one.
		aug_list = [do_nothing, flipud, rot90, rot90_flipud, \
					rot180, rot180_flipud, rot270, rot270_flipud, add_csnt]
		w_aug = [32, 12, 12, 12, 12, 12, 12, 12, 12] # one fourth chances to do_nothing
		transf = choices(aug_list, w_aug)

		# transform all images in array
		return transf[0](sample)

	img_train = datain
	# convert to [N, num_frames*C. H, W] in  [0., 1.] from [N, num_frames, C. H, W] in [0., 255.]
	img_train = img_train.view(img_train.size()[0], -1, \
							   img_train.size()[-2], img_train.size()[-1]) / 255.

	#augment
	img_train = transform(img_train)

	# extract ground truth (central frame)
	gt_train = img_train[:, 3*ctrl_fr_idx:3*ctrl_fr_idx+3, :, :]
	return img_train, gt_train

def init_logging(argdict):
	"""Initilizes the logging and the SummaryWriter modules
	"""
	if not os.path.exists(argdict['log_dir']):
		os.makedirs(argdict['log_dir'])
	writer = SummaryWriter(argdict['log_dir'])
	logger = init_logger(argdict['log_dir'], argdict)
	return writer, logger

def get_imagenames(seq_dir, pattern=None):
	""" Get ordered list of filenames
	"""
	files = []
	for typ in IMAGETYPES:
		files.extend(glob.glob(os.path.join(seq_dir, typ)))

	# filter filenames
	if not pattern is None:
		ffiltered = []
		ffiltered = [f for f in files if pattern in os.path.split(f)[-1]]
		files = ffiltered
		del ffiltered

	# sort filenames alphabetically
	files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
	return files

def open_sequence(seq_dir, gray_mode, expand_if_needed=False, max_num_fr=100):
	r""" Opens a sequence of images and expands it to even sizes if necesary
	Args:
		fpath: string, path to image sequence
		gray_mode: boolean, True indicating if images is to be open are in grayscale mode
		expand_if_needed: if True, the spatial dimensions will be expanded if
			size is odd
		expand_axis0: if True, output will have a fourth dimension
		max_num_fr: maximum number of frames to load
	Returns:
		seq: array of dims [num_frames, C, H, W], C=1 grayscale or C=3 RGB, H and W are even.
			The image gets normalized gets normalized to the range [0, 1].
		expanded_h: True if original dim H was odd and image got expanded in this dimension.
		expanded_w: True if original dim W was odd and image got expanded in this dimension.
	"""
	# Get ordered list of filenames
	files = get_imagenames(seq_dir)

	seq_list = []
	print("\tOpen sequence in folder: ", seq_dir)
	for fpath in files[0:max_num_fr]:

		img, expanded_h, expanded_w = open_image(fpath,\
												   gray_mode=gray_mode,\
												   expand_if_needed=expand_if_needed,\
												   expand_axis0=False)
		seq_list.append(img)
	seq = np.stack(seq_list, axis=0)
	return seq, expanded_h, expanded_w

def open_image(fpath, gray_mode, expand_if_needed=False, expand_axis0=True, normalize_data=True):
	r""" Opens an image and expands it if necesary
	Args:
		fpath: string, path of image file
		gray_mode: boolean, True indicating if image is to be open
			in grayscale mode
		expand_if_needed: if True, the spatial dimensions will be expanded if
			size is odd
		expand_axis0: if True, output will have a fourth dimension
	Returns:
		img: image of dims NxCxHxW, N=1, C=1 grayscale or C=3 RGB, H and W are even.
			if expand_axis0=False, the output will have a shape CxHxW.
			The image gets normalized gets normalized to the range [0, 1].
		expanded_h: True if original dim H was odd and image got expanded in this dimension.
		expanded_w: True if original dim W was odd and image got expanded in this dimension.
	"""
	if not gray_mode:
		# Open image as a CxHxW torch.Tensor
		img = cv2.imread(fpath)
		# from HxWxC to CxHxW, RGB image
		img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
	else:
		# from HxWxC to  CxHxW grayscale image (C=1)
		img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)

	if expand_axis0:
		img = np.expand_dims(img, 0)

	# Handle odd sizes
	expanded_h = False
	expanded_w = False
	sh_im = img.shape
	if expand_if_needed:
		if sh_im[-2]%2 == 1:
			expanded_h = True
			if expand_axis0:
				img = np.concatenate((img, \
					img[:, :, -1, :][:, :, np.newaxis, :]), axis=2)
			else:
				img = np.concatenate((img, \
					img[:, -1, :][:, np.newaxis, :]), axis=1)


		if sh_im[-1]%2 == 1:
			expanded_w = True
			if expand_axis0:
				img = np.concatenate((img, \
					img[:, :, :, -1][:, :, :, np.newaxis]), axis=3)
			else:
				img = np.concatenate((img, \
					img[:, :, -1][:, :, np.newaxis]), axis=2)

	if normalize_data:
		img = normalize(img)
	return img, expanded_h, expanded_w

def batch_psnr(img, imclean, data_range):
	r"""
	Computes the PSNR along the batch dimension (not pixel-wise)

	Args:
		img: a `torch.Tensor` containing the restored image
		imclean: a `torch.Tensor` containing the reference image
		data_range: The data range of the input image (distance between
			minimum and maximum possible values). By default, this is estimated
			from the image data-type.
	"""
	img_cpu = img.data.cpu().numpy().astype(np.float32)
	imgclean = imclean.data.cpu().numpy().astype(np.float32)
	psnr = 0
	for i in range(img_cpu.shape[0]):
		psnr += compare_psnr(imgclean[i, :, :, :], img_cpu[i, :, :, :], \
					   data_range=data_range)
	return psnr/img_cpu.shape[0]

def variable_to_cv2_image(invar, conv_rgb_to_bgr=True):
	r"""Converts a torch.autograd.Variable to an OpenCV image

	Args:
		invar: a torch.autograd.Variable
		conv_rgb_to_bgr: boolean. If True, convert output image from RGB to BGR color space
	Returns:
		a HxWxC uint8 image
	"""
	assert torch.max(invar) <= 1.0

	size4 = len(invar.size()) == 4
	if size4:
		nchannels = invar.size()[1]
	else:
		nchannels = invar.size()[0]

	if nchannels == 1:
		if size4:
			res = invar.data.cpu().numpy()[0, 0, :]
		else:
			res = invar.data.cpu().numpy()[0, :]
		res = (res*255.).clip(0, 255).astype(np.uint8)
	elif nchannels == 3:
		if size4:
			res = invar.data.cpu().numpy()[0]
		else:
			res = invar.data.cpu().numpy()
		res = res.transpose(1, 2, 0)
		res = (res*255.).clip(0, 255).astype(np.uint8)
		if conv_rgb_to_bgr:
			res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
	else:
		raise Exception('Number of color channels not supported')
	return res

def get_git_revision_short_hash():
	r"""Returns the current Git commit.
	"""
	return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()

def init_logger(log_dir, argdict):
	r"""Initializes a logging.Logger to save all the running parameters to a
	log file

	Args:
		log_dir: path in which to save log.txt
		argdict: dictionary of parameters to be logged
	"""
	from os.path import join

	logger = logging.getLogger(__name__)
	logger.setLevel(level=logging.INFO)
	fh = logging.FileHandler(join(log_dir, 'log.txt'), mode='w+')
	formatter = logging.Formatter('%(asctime)s - %(message)s')
	fh.setFormatter(formatter)
	logger.addHandler(fh)
	try:
		logger.info("Commit: {}".format(get_git_revision_short_hash()))
	except Exception as e:
		logger.error("Couldn't get commit number: {}".format(e))
	logger.info("Arguments: ")
	for k in argdict.keys():
		logger.info("\t{}: {}".format(k, argdict[k]))

	return logger

def init_logger_test(result_dir):
	r"""Initializes a logging.Logger in order to log the results after testing
	a model

	Args:
		result_dir: path to the folder with the denoising results
	"""
	from os.path import join

	logger = logging.getLogger('testlog')
	logger.setLevel(level=logging.INFO)
	fh = logging.FileHandler(join(result_dir, 'log.txt'), mode='w+')
	formatter = logging.Formatter('%(asctime)s - %(message)s')
	fh.setFormatter(formatter)
	logger.addHandler(fh)

	return logger

def close_logger(logger):
	'''Closes the logger instance
	'''
	x = list(logger.handlers)
	for i in x:
		logger.removeHandler(i)
		i.flush()
		i.close()

def normalize(data):
	r"""Normalizes a unit8 image to a float32 image in the range [0, 1]

	Args:
		data: a unint8 numpy array to normalize from [0, 255] to [0, 1]
	"""
	return np.float32(data/255.)

def svd_orthogonalization(lyr):
	r"""Applies regularization to the training by performing the
	orthogonalization technique described in the paper "An Analysis and Implementation of
	the FFDNet Image Denoising Method." Tassano et al. (2019).
	For each Conv layer in the model, the method replaces the matrix whose columns
	are the filters of the layer by new filters which are orthogonal to each other.
	This is achieved by setting the singular values of a SVD decomposition to 1.

	This function is to be called by the torch.nn.Module.apply() method,
	which applies svd_orthogonalization() to every layer of the model.
	"""
	classname = lyr.__class__.__name__
	if classname.find('Conv') != -1:
		weights = lyr.weight.data.clone()
		c_out, c_in, f1, f2 = weights.size()
		dtype = lyr.weight.data.type()

		# Reshape filters to columns
		# From (c_out, c_in, f1, f2)  to (f1*f2*c_in, c_out)
		weights = weights.permute(2, 3, 1, 0).contiguous().view(f1*f2*c_in, c_out)

		try:
			# SVD decomposition and orthogonalization
			mat_u, _, mat_v = torch.svd(weights)
			weights = torch.mm(mat_u, mat_v.t())

			lyr.weight.data = weights.view(f1, f2, c_in, c_out).permute(3, 2, 0, 1).contiguous().type(dtype)
		except:
			pass
	else:
		pass

def remove_dataparallel_wrapper(state_dict):
	r"""Converts a DataParallel model to a normal one by removing the "module."
	wrapper in the module dictionary


	Args:
		state_dict: a torch.nn.DataParallel state dictionary
	"""
	from collections import OrderedDict

	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		name = k[7:] # remove 'module.' of DataParallel
		new_state_dict[name] = v

	return new_state_dict

def	resume_training(argdict, model, optimizer):
	""" Resumes previous training or starts anew
	"""
	if argdict['resume_training']:
		resumef = os.path.join(argdict['log_dir'], 'ckpt.pth')
		if os.path.isfile(resumef):
			checkpoint = torch.load(resumef)
			print("> Resuming previous training")
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			new_epoch = argdict['epochs']
			new_milestone = argdict['milestone']
			current_lr = argdict['lr']
			argdict = checkpoint['args']
			training_params = checkpoint['training_params']
			start_epoch = training_params['start_epoch']
			argdict['epochs'] = new_epoch
			argdict['milestone'] = new_milestone
			argdict['lr'] = current_lr
			print("=> loaded checkpoint '{}' (epoch {})"\
				  .format(resumef, start_epoch))
			print("=> loaded parameters :")
			print("==> checkpoint['optimizer']['param_groups']")
			print("\t{}".format(checkpoint['optimizer']['param_groups']))
			print("==> checkpoint['training_params']")
			for k in checkpoint['training_params']:
				print("\t{}, {}".format(k, checkpoint['training_params'][k]))
			argpri = checkpoint['args']
			print("==> checkpoint['args']")
			for k in argpri:
				print("\t{}, {}".format(k, argpri[k]))

			argdict['resume_training'] = False
		else:
			raise Exception("Couldn't resume training with checkpoint {}".\
				   format(resumef))
	else:
		start_epoch = 0
		training_params = {}
		training_params['step'] = 0
		training_params['current_lr'] = 0
		training_params['no_orthog'] = argdict['no_orthog']

	return start_epoch, training_params

def lr_scheduler(epoch, argdict):
	"""Returns the learning rate value depending on the actual epoch number
	By default, the training starts with a learning rate equal to 1e-3 (--lr).
	After the number of epochs surpasses the first milestone (--milestone), the
	lr gets divided by 100. Up until this point, the orthogonalization technique
	is performed (--no_orthog to set it off).
	"""
	# Learning rate value scheduling according to argdict['milestone']
	reset_orthog = False
	if epoch > argdict['milestone'][1]:
		current_lr = argdict['lr'] / 1000.
		reset_orthog = True
	elif epoch > argdict['milestone'][0]:
		current_lr = argdict['lr'] / 10.
	else:
		current_lr = argdict['lr']
	return current_lr, reset_orthog

def	log_train_psnr(result, imsource, loss, writer, epoch, idx, num_minibatches, training_params):

	#Compute pnsr of the whole batch
    psnr_train = batch_psnr(torch.clamp(result, 0., 1.), imsource, 1.)

# Log the scalar values
    writer.add_scalar('loss', loss.item(), training_params['step'])
    writer.add_scalar('PSNR on training data', psnr_train, training_params['step'])
    print("[epoch {}][{}/{}] loss: {:1.4f} PSNR_train: {:1.4f}".format(epoch+1, idx+1, num_minibatches, loss.item(), psnr_train))

def save_model_checkpoint(model, argdict, optimizer, train_pars, epoch):
	"""Stores the model parameters under 'argdict['log_dir'] + '/net.pth'
	Also saves a checkpoint under 'argdict['log_dir'] + '/ckpt.pth'
	"""
	torch.save(model.state_dict(), os.path.join(argdict['log_dir'], 'net.pth'))
	save_dict = { \
		'state_dict': model.state_dict(), \
		'optimizer' : optimizer.state_dict(), \
		'training_params': train_pars, \
		'args': argdict\
		}
	torch.save(save_dict, os.path.join(argdict['log_dir'], 'ckpt.pth'))

	if epoch % argdict['save_every_epochs'] == 0:
		torch.save(save_dict, os.path.join(argdict['log_dir'], 'ckpt_e{}.pth'.format(epoch+1)))
	del save_dict

def validate_and_log(model_temp, dataset_val, valnoisestd, temp_psz, writer, \
					 epoch, lr, logger, trainimg):
	"""Validation step after the epoch finished
	"""
	t1 = time.time()
	psnr_val = 0
	with torch.no_grad():
		for seq_val in dataset_val:
			noise = torch.FloatTensor(seq_val.size()).normal_(mean=0, std=valnoisestd)
			seqn_val = seq_val + noise
			seqn_val = seqn_val.cuda()
			sigma_noise = torch.cuda.FloatTensor([valnoisestd])
			out_val = denoise_seq_fastdvdnet(seq=seqn_val, \
											noise_std=sigma_noise, \
											temp_psz=temp_psz,\
											model_temporal=model_temp)
			psnr_val += batch_psnr(out_val.cpu(), seq_val.squeeze_(), 1.)
		psnr_val /= len(dataset_val)
		t2 = time.time()
		print("\n[epoch %d] PSNR_val: %.4f, on %.2f sec" % (epoch+1, psnr_val, (t2-t1)))
		writer.add_scalar('PSNR on validation data', psnr_val, epoch)
		writer.add_scalar('Learning rate', lr, epoch)

	# Log val images
	try:
		idx = 0
		if epoch == 0:

			# Log training images
			_, _, Ht, Wt = trainimg.size()
			img = tutils.make_grid(trainimg.view(-1, 3, Ht, Wt), \
								   nrow=8, normalize=True, scale_each=True)
			writer.add_image('Training patches', img, epoch)

			# Log validation images
			img = tutils.make_grid(seq_val.data[idx].clamp(0., 1.),\
									nrow=2, normalize=False, scale_each=False)
			imgn = tutils.make_grid(seqn_val.data[idx].clamp(0., 1.),\
									nrow=2, normalize=False, scale_each=False)
			writer.add_image('Clean validation image {}'.format(idx), img, epoch)
			writer.add_image('Noisy validation image {}'.format(idx), imgn, epoch)

		# Log validation results
		irecon = tutils.make_grid(out_val.data[idx].clamp(0., 1.),\
								nrow=2, normalize=False, scale_each=False)
		writer.add_image('Reconstructed validation image {}'.format(idx), irecon, epoch)

	except Exception as e:
		logger.error("validate_and_log_temporal(): Couldn't log results, {}".format(e))

def temp_denoise(model, noisyframe, sigma_noise):
	'''Encapsulates call to denoising model and handles padding.
		Expects noisyframe to be normalized in [0., 1.]
	'''
	# make size a multiple of four (we have two scales in the denoiser)
	sh_im = noisyframe.size()
	expanded_h = sh_im[-2]%4
	if expanded_h:
		expanded_h = 4-expanded_h
	expanded_w = sh_im[-1]%4
	if expanded_w:
		expanded_w = 4-expanded_w
	padexp = (0, expanded_w, 0, expanded_h)
	noisyframe = F.pad(input=noisyframe, pad=padexp, mode='reflect')
	sigma_noise = F.pad(input=sigma_noise, pad=padexp, mode='reflect')

	# denoise
	out = torch.clamp(model(noisyframe, sigma_noise), 0., 1.)

	if expanded_h:
		out = out[:, :, :-expanded_h, :]
	if expanded_w:
		out = out[:, :, :, :-expanded_w]

	return out

def denoise_seq_fastdvdnet(seq, noise_std, temp_psz, model_temporal):
	r"""Denoises a sequence of frames with FastDVDnet.

	Args:
		seq: Tensor. [numframes, 1, C, H, W] array containing the noisy input frames
		noise_std: Tensor. Standard deviation of the added noise
		temp_psz: size of the temporal patch
		model_temp: instance of the PyTorch model of the temporal denoiser
	Returns:
		denframes: Tensor, [numframes, C, H, W]
	"""
	# init arrays to handle contiguous frames and related patches
	numframes, C, H, W = seq.shape
	ctrlfr_idx = int((temp_psz-1)//2)
	inframes = list()
	denframes = torch.empty((numframes, C, H, W)).to(seq.device)

	# build noise map from noise std---assuming Gaussian noise
	noise_map = noise_std.expand((1, 1, H, W))

	for fridx in range(numframes):
		# load input frames
		if not inframes:
		# if list not yet created, fill it with temp_patchsz frames
			for idx in range(temp_psz):
				relidx = abs(idx-ctrlfr_idx) # handle border conditions, reflect
				inframes.append(seq[relidx])
		else:
			del inframes[0]
			relidx = min(fridx + ctrlfr_idx, -fridx + 2*(numframes-1)-ctrlfr_idx) # handle border conditions
			inframes.append(seq[relidx])

		inframes_t = torch.stack(inframes, dim=0).contiguous().view((1, temp_psz*C, H, W)).to(seq.device)

		# append result to output list
		denframes[fridx] = temp_denoise(model_temporal, inframes_t, noise_map)

	# free memory up
	del inframes
	del inframes_t
	torch.cuda.empty_cache()

	# convert to appropiate type and return
	return denframes

