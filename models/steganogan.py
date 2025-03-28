# -*- coding: utf-8 -*-
import gc
import inspect
import json
import os
from collections import Counter

import imageio
import torch
from imageio import imread, imwrite
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torch.optim import Adam
from tqdm import tqdm

from steganogan.utils import bits_to_bytearray, bytearray_to_text, psnr, ssim, text_to_bits

DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'train')

METRIC_FIELDS = [
    'val.encoder_mse',
    'val.decoder_loss',
    'val.decoder_acc',
    'val.cover_score',
    'val.generated_score',
    'val.ssim',
    'val.psnr',
    'val.bpp',
    'train.encoder_mse',
    'train.decoder_loss',
    'train.decoder_acc',
    'train.cover_score',
    'train.generated_score',
]


class SteganoGAN(object):
    """SteganoGAN: A GAN-based steganography tool.

    The SteganoGAN class coordinates the training and inference processes
    for the steganography model, including encoding messages into images
    and decoding them back.

    Args:
        data_depth (int):
            Number of channels for hidden data.
        encoder (class or instance):
            Encoder network or class to use.
        decoder (class or instance):
            Decoder network or class to use.
        critic (class or instance):
            Critic network or class to use.
        cuda (bool):
            Whether to use CUDA for training and inference.
        verbose (bool):
            Whether to display progress information.
        log_dir (str):
            Directory to save logs and samples.
        **kwargs:
            Additional arguments for the encoder, decoder and critic networks.
    """

    def _get_instance(self, class_or_instance, kwargs):
        """Returns an instance of the class."""

        if not inspect.isclass(class_or_instance):
            return class_or_instance

        argspec = inspect.getfullargspec(class_or_instance.__init__).args
        argspec.remove('self')
        init_args = {arg: kwargs[arg] for arg in argspec if arg in kwargs}

        return class_or_instance(**init_args)

    def set_device(self, cuda=True):
        """Sets the torch device depending on whether cuda is avaiable or not."""
        if cuda and torch.cuda.is_available():
            self.cuda = True
            self.device = torch.device('cuda')
        else:
            self.cuda = False
            self.device = torch.device('cpu')

        if self.verbose:
            if not cuda:
                print('Using CPU device')
            elif not self.cuda:
                print('CUDA is not available. Defaulting to CPU device')
            else:
                print('Using CUDA device')

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.critic.to(self.device)

    def __init__(self, data_depth, encoder, decoder, critic,
                 cuda=False, verbose=False, log_dir=None, **kwargs):
        """Initialize a new instance of SteganoGAN."""
        
        self.verbose = verbose

        self.data_depth = data_depth
        kwargs['data_depth'] = data_depth
        self.encoder = self._get_instance(encoder, kwargs)
        self.decoder = self._get_instance(decoder, kwargs)
        self.critic = self._get_instance(critic, kwargs)
        self.set_device(cuda)

        self.critic_optimizer = None
        self.decoder_optimizer = None

        # Misc
        self.fit_metrics = None
        self.history = list()

        self.log_dir = log_dir
        if log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self.samples_path = os.path.join(self.log_dir, 'samples')
            os.makedirs(self.samples_path, exist_ok=True)

    def _random_data(self, cover):
        """Generate random data to be hidden inside the cover image.

        Args:
            cover (torch.Tensor): 
                Image to use as cover.

        Returns:
            torch.Tensor: 
                Random binary data tensor.
        """
        N, _, H, W = cover.size()
        return torch.zeros((N, self.data_depth, H, W), device=self.device).random_(0, 2)

    def _encode_decode(self, cover, quantize=False):
        """Encode random data and then decode it.

        Args:
            cover (torch.Tensor): 
                Image to use as cover.
            quantize (bool): 
                Whether to quantize the generated image or not.

        Returns:
            tuple (torch.Tensor, torch.Tensor, torch.Tensor): 
                Generated image, payload data, decoded data.
        """
        payload = self._random_data(cover)
        generated = self.encoder(cover, payload)
        if quantize:
            generated = (255.0 * (generated + 1.0) / 2.0).long()
            generated = 2.0 * generated.float() / 255.0 - 1.0

        decoded = self.decoder(generated)

        return generated, payload, decoded

    def _critic(self, image):
        """Evaluate the image using the critic.

        Args:
            image (torch.Tensor): 
                Image to evaluate.

        Returns:
            torch.Tensor: 
                Critic score.
        """
        return torch.mean(self.critic(image))

    def _get_optimizers(self):
        """Get optimizers for the decoder and critic.

        Returns:
            tuple (torch.optim.Adam, torch.optim.Adam): 
                Critic optimizer, decoder optimizer.
        """
        _dec_list = list(self.decoder.parameters()) + list(self.encoder.parameters())
        critic_optimizer = Adam(self.critic.parameters(), lr=1e-4)
        decoder_optimizer = Adam(_dec_list, lr=1e-4)

        return critic_optimizer, decoder_optimizer

    def _fit_critic(self, train, metrics):
        """Train the critic to distinguish between cover and steganographic images.

        Args:
            train (torch.utils.data.DataLoader): 
                Training data loader.
            metrics (dict): 
                Dictionary to store metrics.
        """
        for cover, _ in tqdm(train, disable=not self.verbose):
            gc.collect()
            cover = cover.to(self.device)
            payload = self._random_data(cover)
            generated = self.encoder(cover, payload)
            cover_score = self._critic(cover)
            generated_score = self._critic(generated)

            self.critic_optimizer.zero_grad()
            (cover_score - generated_score).backward(retain_graph=False)
            self.critic_optimizer.step()

            for p in self.critic.parameters():
                p.data.clamp_(-0.1, 0.1)

            metrics['train.cover_score'].append(cover_score.item())
            metrics['train.generated_score'].append(generated_score.item())

    def _fit_coders(self, train, metrics):
        """Train the encoder and decoder on the training images.

        Args:
            train (torch.utils.data.DataLoader): 
                Training data loader.
            metrics (dict): 
                Dictionary to store metrics.
        """
        for cover, _ in tqdm(train, disable=not self.verbose):
            gc.collect()
            cover = cover.to(self.device)
            generated, payload, decoded = self._encode_decode(cover)
            encoder_mse, decoder_loss, decoder_acc = self._coding_scores(
                cover, generated, payload, decoded)
            generated_score = self._critic(generated)

            self.decoder_optimizer.zero_grad()
            (100.0 * encoder_mse + decoder_loss + generated_score).backward()
            self.decoder_optimizer.step()

            metrics['train.encoder_mse'].append(encoder_mse.item())
            metrics['train.decoder_loss'].append(decoder_loss.item())
            metrics['train.decoder_acc'].append(decoder_acc.item())

    def _coding_scores(self, cover, generated, payload, decoded):
        """Calculate scores for the encoder and decoder.

        Args:
            cover (torch.Tensor): 
                Cover image.
            generated (torch.Tensor): 
                Generated steganographic image.
            payload (torch.Tensor): 
                Payload data.
            decoded (torch.Tensor): 
                Decoded data.

        Returns:
            tuple (torch.Tensor, torch.Tensor, torch.Tensor): 
                Encoder MSE, decoder loss, decoder accuracy.
        """
        encoder_mse = mse_loss(generated, cover)
        decoder_loss = binary_cross_entropy_with_logits(decoded, payload)
        decoder_acc = (decoded >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()

        return encoder_mse, decoder_loss, decoder_acc

    def _validate(self, validate, metrics):
        """Validate the model on the validation dataset.

        Args:
            validate (torch.utils.data.DataLoader): 
                Validation data loader.
            metrics (dict): 
                Dictionary to store metrics.
        """
        for cover, _ in tqdm(validate, disable=not self.verbose):
            gc.collect()
            cover = cover.to(self.device)
            generated, payload, decoded = self._encode_decode(cover, quantize=True)
            encoder_mse, decoder_loss, decoder_acc = self._coding_scores(
                cover, generated, payload, decoded)
            generated_score = self._critic(generated)
            cover_score = self._critic(cover)

            metrics['val.encoder_mse'].append(encoder_mse.item())
            metrics['val.decoder_loss'].append(decoder_loss.item())
            metrics['val.decoder_acc'].append(decoder_acc.item())
            metrics['val.cover_score'].append(cover_score.item())
            metrics['val.generated_score'].append(generated_score.item())
            metrics['val.ssim'].append(ssim(cover, generated).item())
            metrics['val.psnr'].append(psnr(cover, generated).item())
            metrics['val.bpp'].append(self.data_depth * (2 * decoder_acc.item() - 1))

    def _generate_samples(self, samples_path, cover, epoch):
        """Generate sample images for logging.

        Args:
            samples_path (str): 
                Path to save the samples.
            cover (torch.Tensor): 
                Cover image.
            epoch (int): 
                Current epoch.
        """
        cover = cover.to(self.device)
        generated, payload, decoded = self._encode_decode(cover)
        samples = generated.size(0)
        for sample in range(samples):
            cover_path = os.path.join(samples_path, '{}.cover.png'.format(sample))
            sample_name = '{}.generated-{:2d}.png'.format(sample, epoch)
            sample_path = os.path.join(samples_path, sample_name)

            image = (cover[sample].permute(1, 2, 0).detach().cpu().numpy() + 1.0) / 2.0
            imageio.imwrite(cover_path, (255.0 * image).astype('uint8'))

            sampled = generated[sample].clamp(-1.0, 1.0).permute(1, 2, 0)
            sampled = sampled.detach().cpu().numpy() + 1.0

            image = sampled / 2.0
            imageio.imwrite(sample_path, (255.0 * image).astype('uint8'))

    def fit(self, train, validate, epochs=5):
        """Train a new model with the given data loaders.

        Args:
            train (torch.utils.data.DataLoader): 
                Training data loader.
            validate (torch.utils.data.DataLoader): 
                Validation data loader.
            epochs (int): 
                Number of epochs to train for.
        """
        if self.critic_optimizer is None:
            self.critic_optimizer, self.decoder_optimizer = self._get_optimizers()
            self.epochs = 0

        if self.log_dir:
            sample_cover = next(iter(validate))[0]

        # Start training
        total = self.epochs + epochs
        for epoch in range(1, epochs + 1):
            # Count how many epochs we have trained for this steganogan
            self.epochs += 1

            metrics = {field: list() for field in METRIC_FIELDS}

            if self.verbose:
                print('Epoch {}/{}'.format(self.epochs, total))

            self._fit_critic(train, metrics)
            self._fit_coders(train, metrics)
            self._validate(validate, metrics)

            self.fit_metrics = {k: sum(v) / len(v) for k, v in metrics.items() if v}
            self.fit_metrics['epoch'] = epoch

            if self.log_dir:
                self.history.append(self.fit_metrics)

                metrics_path = os.path.join(self.log_dir, 'metrics.log')
                with open(metrics_path, 'w') as metrics_file:
                    json.dump(self.history, metrics_file, indent=4)

                save_name = '{}.bpp-{:03f}.p'.format(
                    self.epochs, self.fit_metrics['val.bpp'])

                self.save(os.path.join(self.log_dir, save_name))
                self._generate_samples(self.samples_path, sample_cover, epoch)

            # Empty cuda cache (this may help for memory leaks)
            if self.cuda:
                torch.cuda.empty_cache()

            gc.collect()

    def _make_payload(self, width, height, depth, text):
        """Create a payload tensor from text.

        This takes a piece of text and encodes it into a bit vector. It then
        fills a matrix of size (width, height) with copies of the bit vector.

        Args:
            width (int): 
                Width of the payload tensor.
            height (int): 
                Height of the payload tensor.
            depth (int): 
                Depth of the payload tensor.
            text (str): 
                Text to encode.

        Returns:
            torch.Tensor: 
                Payload tensor.
        """
        message = text_to_bits(text) + [0] * 32

        payload = message
        while len(payload) < width * height * depth:
            payload += message

        payload = payload[:width * height * depth]

        return torch.FloatTensor(payload).view(1, depth, height, width)

    def encode(self, cover, output, text):
        """Encode an image with a message.

        Args:
            cover (str): 
                Path to the cover image.
            output (str): 
                Path where the generated image will be saved.
            text (str): 
                Message to hide inside the image.
        """
        cover = imread(cover, pilmode='RGB') / 127.5 - 1.0
        cover = torch.FloatTensor(cover).permute(2, 0, 1).unsqueeze(0)

        cover_size = cover.size()
        payload = self._make_payload(cover_size[3], cover_size[2], self.data_depth, text)

        cover = cover.to(self.device)
        payload = payload.to(self.device)
        generated = self.encoder(cover, payload)[0].clamp(-1.0, 1.0)

        generated = (generated.permute(1, 2, 0).detach().cpu().numpy() + 1.0) * 127.5
        imwrite(output, generated.astype('uint8'))

        if self.verbose:
            print('Encoding completed.')

    def decode(self, image):
        """Decode a message from an image.

        Args:
            image (str): 
                Path to the image with the hidden message.

        Returns:
            str: 
                Decoded message.

        Raises:
            ValueError: 
                If unable to read the image or find a message.
        """
        if not os.path.exists(image):
            raise ValueError('Unable to read image: {}'.format(image))

        # extract a bit vector
        image = imread(image, pilmode='RGB') / 255.0
        image = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)
        image = image.to(self.device)

        image = self.decoder(image).view(-1) > 0

        # split and decode messages
        candidates = Counter()
        bits = image.data.cpu().numpy().tolist()
        for candidate in bits_to_bytearray(bits).split(b'\x00\x00\x00\x00'):
            candidate = bytearray_to_text(bytearray(candidate))
            if candidate:
                candidates[candidate] += 1

        # choose most common message
        if len(candidates) == 0:
            raise ValueError('Failed to find message.')

        candidate, count = candidates.most_common(1)[0]
        return candidate

    def save(self, path):
        """Save the model to a file.

        Args:
            path (str): 
                Path to save the model.
        """
        torch.save(self, path)

    @classmethod
    def load(cls, architecture=None, path=None, cuda=True, verbose=False):
        """Load a pre-trained model.

        Args:
            architecture (str): 
                Name of a pretrained model to load.
            path (str): 
                Path to a saved model.
            cuda (bool): 
                Whether to use CUDA.
            verbose (bool): 
                Whether to display progress information.

        Returns:
            SteganoGAN: 
                Loaded model.

        Raises:
            ValueError: 
                If neither architecture nor path is provided, or if both are provided.
        """
        if architecture and not path:
            model_name = '{}.steg'.format(architecture)
            pretrained_path = os.path.join(os.path.dirname(__file__), '..', 'pretrained')
            path = os.path.join(pretrained_path, model_name)

        elif (architecture is None and path is None) or (architecture and path):
            raise ValueError(
                'Please provide either an architecture or a path to pretrained model.')

        steganogan = torch.load(path, map_location='cpu')
        steganogan.verbose = verbose

        steganogan.encoder.upgrade_legacy()
        steganogan.decoder.upgrade_legacy()
        steganogan.critic.upgrade_legacy()

        steganogan.set_device(cuda)
        return steganogan