import os
import torch
import pytorch_lightning as pl

import config
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from tcnn_module import TCNN
from LARA_data_module import LARADataModule


torch.set_float32_matmul_precision("medium") # to make lightning happy

if __name__ == "__main__":
    MONITOR_LOSS = "validation_loss"
    CHECKPOINT_PATH = "pl_results/"

    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)


    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor=MONITOR_LOSS,
        mode="min",
        dirpath=CHECKPOINT_PATH,
        filename="checkpoint-TCNN-{epoch:02d}-{val_loss:.2f}",
    )    

    logger = TensorBoardLogger("tb_logs", name="TCNN-logs")
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler0"),
        schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20),
    )

    model = TCNN(
        learning_rate=config.LEARNING_RATE,
        num_filters=config.NUM_FILTERS,
        filter_size=config.FILTER_SIZE,
        mode=config.MODE,
        num_attributes=config.NUM_ATTRIBUTES,
        num_classes=config.NUM_CLASSES,
        window_length=config.WINDOW_LENGTH,
        sensor_channels=config.NUM_SENSORS,
        path_attributes=config.PATH_ATTRIBUTES,
        
    )

    data_module = LARADataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        accelerator=config.ACCELERATOR,
    )
    
    trainer = pl.Trainer(
        accelerator=config.ACCELERATOR,
        profiler=profiler,
        logger=logger,
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        precision=config.PRECISION,
        # limit_val_batches=1000,
        val_check_interval=config.VAL_BATCHES,
        callbacks=[EarlyStopping(monitor=MONITOR_LOSS),checkpoint_callback],
    )
    trainer.fit(model, data_module)
    trainer.validate(model, data_module)
    trainer.test(model, data_module)