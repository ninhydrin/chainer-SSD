import time
import datetime
import sys
import os

class Logger:
    def __init__(self, f_name, res_q, args):
        self.f_name = f_name
        self.res_q = res_q
        self.args = args
        if not os.path.isdir("log"):
            os.mkdir("log")

    def out_log(self, log_str, write_type="a"):
        with open(self.f_name, write_type) as f:
            f.write(log_str)

    def __call__(self):
        # Logger
        self.out_log("start train {}\n".format(
            datetime.datetime.now().isoformat()), "w")
        args = self.args
        epoch = 0
        train_count = 0
        train_cur_loss = 0
        train_cur_accuracy = 0
        begin_at = time.time()
        val_begin_at = None
        while True:
            result = self.res_q.get()
            if result == 'end':
                print(file=sys.stderr)
                break
            elif result == '\nepoch\n':
                epoch += 1
                print("epoch : {}".format(epoch))
                continue
            elif result == 'train':
                print(file=sys.stderr)
                train = True
                if val_begin_at is not None:
                    begin_at += time.time() - val_begin_at
                    val_begin_at = None
                continue
            elif result == 'val':
                print(file=sys.stderr)
                train = False
                val_count = val_loss = val_accuracy = 0
                val_begin_at = time.time()
                continue

            loss, accuracy = result
            if train:
                train_count += 1
                duration = time.time() - begin_at
                throughput = train_count * args.batchsize / duration
                sys.stderr.write(
                    '\rtrain {0} updates ({1} samples) time: {2} ({3:.4} fps) loss={4:.4} '
                    .format(train_count, train_count * args.batchsize,
                            datetime.timedelta(seconds=duration), throughput,loss))

                train_cur_loss += loss
                train_cur_accuracy += accuracy

                if train_count % 1000 == 0:
                    mean_loss = train_cur_loss / 1000
                    mean_error = 1 - train_cur_accuracy / 1000
                    log_str = "type : {}, iter : {}, error : {}, loss : {}\n".format("train", train_count, mean_error, mean_loss)
                    print(file=sys.stderr)
                    print(log_str)
                    sys.stdout.flush()
                    train_cur_loss = 0
                    train_cur_accuracy = 0
            else:
                val_count += args.val_batchsize
                duration = time.time() - val_begin_at
                throughput = val_count / duration
                sys.stderr.write(
                    '\rval   {0} batches ({1} samples) time: {2} ({3:.4} images/sec) loss = {4:.4}'
                    .format(val_count / args.val_batchsize, val_count,
                            datetime.timedelta(seconds=duration), throughput, loss))

                val_loss += loss
                val_accuracy += accuracy
                if val_count == 50000:
                    mean_loss = val_loss * args.val_batchsize / 50000
                    mean_error = 1 - val_accuracy * args.val_batchsize / 50000
                    log_str = "type : {}, iter : {}, error : {}, loss : {}\n".format("val", train_count, mean_error, mean_loss)
                    print(file=sys.stderr)
                    print(log_str)
                    self.out_log(log_str)
                    sys.stdout.flush()
