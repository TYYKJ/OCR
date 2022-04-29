from ocr import CRNNTrainer

if __name__ == '__main__':
    m = CRNNTrainer(
        project_name='hs512CycleLR',
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
        batch_size=16,
        num_workers=16,
        optimizer_name='sgd',
        lr=0.01,
        weight_decay=1e-4,
        momentum=0.9,
        # resume_path='../weights/hs512CycleLR-CRNN-resnet101vd-epoch=73-val_acc=0.90.ckpt',
        use_augmentation=True,
        encoder_type='reshape',
        hidden_size=512
    )

    m.build_trainer(
        gpus=[0],
        epochs=100,
        precision=16,
        benchmark=True,
    )
