from utils import *
def run_epoch(data_iter,
             model,
             loss_compute,
             optimizer,
             scheduler,
             mode = "train",
             accum_iter = 1,
             train_state = TrainState(),):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    #out1 = torch.tensor([[1]], device = device)
    #out1.to(device)
    for i , batch in enumerate(data_iter):
        
        out = model.forward(batch.src, batch.tgt,
                           batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        loss_node.backward()
        
        train_state.step += 1
        train_state.samples += batch.src.shape[0]
        train_state.tokens += batch.ntokens        
        
        optimizer.step()
        optimizer.zero_grad()
        n_accum += 1
        train_state.accum_step += 1
        scheduler.step()
        
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 100 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
        del loss
        del loss_node
    return total_loss / total_tokens , train_state


def train_worker(model,ngpus_per_node,
                vocab_src, vocab_tgt,
                spacy_en, spacy_hi,
                config, optimizer, lr_scheduler, is_distributed = False,):
    device = torch.device("cuda" if torch.cuda.is_available else 'cpu')
    pad_idx = vocab_tgt["<blank>"]
    d_model = 512
    model = model
    model.to(device)
    module = model
    is_main_process = True
    criterion = LabelSmoothing(size = len(vocab_tgt), padding_idx = pad_idx, smoothing = 0.1)
    criterion.to(device)
    train_dataloader = create_dataloaders(device, vocab_src, vocab_tgt,
                                        spacy_en, spacy_hi, batch_size = config["batch_size"],
                                       max_padding = config["max_padding"],
                                       is_distributed = is_distributed,)
    
    
    train_state = TrainState()
    model.train()
    for epoch in range(config["num_epochs"]):
        
        _, train_state = run_epoch((Batch(b[0], b[1], pad_idx) for b in train_dataloader),
                                  model,
                                  SimpleLossCompute(model.generator, criterion),
                                  optimizer,
                                  lr_scheduler,
                                  mode = "train+log",
                                  accum_iter = config["accum_iter"],
                                  train_state = train_state,)   

def train_model(model, vocab_src, vocab_tgt, spacy_en, spacy_hi, config, optimizer, lr_scheduler):
    train_worker(model,1, vocab_src, vocab_tgt, spacy_en, spacy_hi, config, optimizer, lr_scheduler, False)
