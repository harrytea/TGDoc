from transformers import Trainer


class TGDocTrainer(Trainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        super(TGDocTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir=None, state_dict=None):
        super(TGDocTrainer, self)._save(output_dir, state_dict)
