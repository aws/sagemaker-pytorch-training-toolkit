from fastai.script import *
from fastai.vision import *

@call_parse
def main():
    tgz_path = os.environ.get('SM_CHANNEL_TRAINING')
    path = os.path.join(tgz_path, 'mnist_tiny')
    tarfile.open(f'{path}.tgz', 'r:gz').extractall(tgz_path)
    tfms = (rand_pad(2, 28), [])
    data = ImageDataBunch.from_folder(path, ds_tfms=tfms, bs=64)
    data.normalize(imagenet_stats)
    learn = create_cnn(data, models.resnet18, metrics=accuracy, path='/opt/ml', model_dir='model')
    learn.fit_one_cycle(1, 0.02)
    learn.save(name='model')
