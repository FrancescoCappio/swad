import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from domainbed.lib.fast_data_loader import FastDataLoader

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def accuracy_from_loader(algorithm, loader, weights, debug=False, open_set=False, known_classes=None):
    correct = 0
    total = 0
    losssum = 0.0
    weights_offset = 0

    algorithm.eval()

    predictions = {}

    for i, batch in enumerate(loader):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        
        if 'idx' in batch:
            ids = batch['idx']

        with torch.no_grad():
            logits = algorithm.predict(x)
            if open_set:
                logits=logits[:,:-1] # esclude last class (unknown)
                preds=nn.Softmax(dim=1)(logits)
                for i, img_index in enumerate(ids): 
                    predictions[img_index.item()]={'cls': preds[i].argmax().cpu().item(), 'conf': preds[i].max().cpu().item()}
                mask = y < known_classes
                x = x[mask]
                y = y[mask]
                logits = logits[mask]

            loss = F.cross_entropy(logits, y).item()


        B = len(x)
        losssum += loss * B

        if weights is None:
            batch_weights = torch.ones(len(x))
        else:
            batch_weights = weights[weights_offset : weights_offset + len(x)]
            weights_offset += len(x)
        batch_weights = batch_weights.to(device)
        if logits.size(1) == 1:
            correct += (logits.gt(0).eq(y).float() * batch_weights).sum().item()
        else:
            correct += (logits.argmax(1).eq(y).float() * batch_weights).sum().item()
        total += batch_weights.sum().item()

        if debug:
            break

    algorithm.train()

    acc = correct / total
    loss = losssum / total
    return acc, loss, correct, total, predictions


def accuracy(algorithm, loader_kwargs, weights, **kwargs):
    if isinstance(loader_kwargs, dict):
        loader = FastDataLoader(**loader_kwargs)
    elif isinstance(loader_kwargs, FastDataLoader):
        loader = loader_kwargs
    else:
        raise ValueError(loader_kwargs)
    return accuracy_from_loader(algorithm, loader, weights, **kwargs)


class Evaluator:
    def __init__(
        self, test_envs, eval_meta, n_envs, logger, evalmode="fast", debug=False, target_env=None, open_set=False, known_classes=None,
    ):
        all_envs = list(range(n_envs))
        train_envs = sorted(set(all_envs) - set(test_envs))
        self.test_envs = test_envs
        self.train_envs = train_envs
        self.eval_meta = eval_meta
        self.n_envs = n_envs
        self.logger = logger
        self.evalmode = evalmode
        self.debug = debug
        self.open_set = open_set
        self.known_classes = known_classes

        if target_env is not None:
            self.set_target_env(target_env)

    def set_target_env(self, target_env):
        """When len(test_envs) == 2, you can specify target env for computing exact test acc."""
        self.test_envs = [target_env]

    def evaluate(self, algorithm, ret_losses=False, ret_predictions=False):
        n_train_envs = len(self.train_envs)
        n_test_envs = len(self.test_envs)
        assert n_test_envs == 1
        summaries = collections.defaultdict(float)
        # for key order
        summaries["test_in"] = 0.0
        summaries["test_out"] = 0.0
        summaries["test_all"] = 0.0
        summaries["train_in"] = 0.0
        summaries["train_out"] = 0.0
        accuracies = {}
        losses = {}
        test_correct_all = 0
        test_count_all = 0
        test_predictions = {}

        # order: in_splits + out_splits.
        for name, loader_kwargs, weights in self.eval_meta:
            # env\d_[in|out]
            env_name, inout = name.split("_")
            env_num = int(env_name[3:])

            skip_eval = self.evalmode == "fast" and inout == "in" and env_num not in self.test_envs
            if skip_eval:
                continue

            is_test = env_num in self.test_envs
            acc, loss, correct, total, predictions = accuracy(algorithm, loader_kwargs, weights, debug=self.debug,open_set=self.open_set,known_classes=self.known_classes)
            accuracies[name] = acc
            losses[name] = loss

            if env_num in self.train_envs:
                summaries["train_" + inout] += acc / n_train_envs
                if inout == "out":
                    summaries["tr_" + inout + "loss"] += loss / n_train_envs
            elif is_test:
                summaries["test_" + inout] += acc / n_test_envs
                test_correct_all += correct
                test_count_all += total
                test_predictions.update(predictions)

        summaries["test_all"] = test_correct_all/ test_count_all
        if ret_losses:
            return accuracies, summaries, losses
        else:
            if ret_predictions:
                return accuracies, summaries, test_predictions
            return accuracies, summaries
