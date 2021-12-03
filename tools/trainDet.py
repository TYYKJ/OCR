from ocr.det import DBTrainer

m = DBTrainer(
    train_data_path='/home/cat/Documents/ICDAR/ICDAR2019/train.json',
    val_data_path='/home/cat/Documents/ICDAR/ICDAR2017/val.json',
    checkpoint_save_path='../weights',
    encoder_name='dpn68',
    weights='imagenet',
    lr=0.01,
    batch_size=16,
    num_workers=16,
    optimizer_name='sgd',
    # resume_path='../weights/DB-dpn68-epoch=70-hmean=0.43-recall=0.35-precision=0.56.ckpt'
)
m.build_trainer(
    gpus=[1],
    max_epochs=100,
)
