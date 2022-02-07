from ocr import CRNNTrainer

if __name__ == '__main__':
    m = CRNNTrainer(
        project_name='ICDAR',
        encoder_name='resnet18vd',
        image_path='/home/cat/Documents/icdar2015-ok/recognition/image',
        train_label_path='/home/cat/Documents/icdar2015-ok/recognition/train-no-space.txt',
        val_label_path='/home/cat/Documents/icdar2015-ok/recognition/test-no-space.txt',
        checkpoint_save_path='../weights',
        classes=83 + 1,
        alphabet_path='/home/cat/PycharmProjects/OCR/makeDataset/dict.txt',
        input_h=32,
        mean=0.5,
        std=0.5,
        batch_size=8,
        num_workers=16,
        optimizer_name='sgd',
        lr=0.001,
        weight_decay=0.,
        momentum=0.9,
        # resume_path='../weights/CRNN-resnet50vd-epoch=00-val_acc=0.18.ckpt',
        use_augmentation=False
    )

    m.build_trainer(
        gpus=[1],
        max_epochs=1200,
    )
