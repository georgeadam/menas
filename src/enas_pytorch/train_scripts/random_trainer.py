"""The module for training ENAS shared params only with a random controller."""
import utils

from train_scripts.regular_trainer import Trainer, _get_no_grad_ctx_mgr

logger = utils.get_logger()


class RandomTrainer(Trainer):
    """A class to wrap training code."""

    def __init__(self, args, dataset):
        """Constructor for training algorithm.

        Args:
            args: From command line, picked up by `argparse`.
            dataset: Currently only `data.text.Corpus` is supported.

        Initializes:
            - Data: train, val and test.
            - Model: shared and controller.
            - Inference: optimizers for shared and controller parameters.
            - Criticism: cross-entropy loss for training the shared model.
        """
        super().__init__(args, dataset)

    def train(self):
        """Cycles through alternately training the shared parameters and the
        controller, as described in Section 2.2, Training ENAS and Deriving
        Architectures, of the paper.

        From the paper (for Penn Treebank):

        - Just the shared parameters are trained since we want to sample from a random controller.
        """
        if self.args.shared_initial_step > 0:
            self.train_shared(self.args.shared_initial_step)

        for self.epoch in range(self.start_epoch, self.args.max_epoch):
            # 1. Training the shared parameters omega of the child models
            self.train_shared()

            # 2. Training the controller parameters theta

            if self.epoch % self.args.save_epoch == 0:
                with _get_no_grad_ctx_mgr():
                    best_dag = self.derive()
                    eval_ppl = self.evaluate(self.eval_data,
                                  best_dag,
                                  'val_best',
                                  max_num=self.args.batch_size) # * 100

                if eval_ppl < self.best_ppl:
                    self.best_ppl = eval_ppl
                    self.save_model(self.shared_path, self.controller_path)

            if self.epoch >= self.args.shared_decay_after:
                utils.update_lr(self.shared_optim, self.shared_lr)

        self.load_model()
        self.test()