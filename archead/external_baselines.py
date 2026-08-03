import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@contextmanager
def suppress_output():
    if "--verbose" in sys.argv:
        yield
        return

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    try:
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
        
        saved_stdout_fd = os.dup(stdout_fd)
        saved_stderr_fd = os.dup(stderr_fd)
        
        devnull = open(os.devnull, "wb")
        os.dup2(devnull.fileno(), stdout_fd)
        os.dup2(devnull.fileno(), stderr_fd)
    except Exception:
        stdout_fd = None
        stderr_fd = None
        devnull = open(os.devnull, "w")
        sys.stdout = devnull
        sys.stderr = devnull

    try:
        yield
    finally:
        if stdout_fd is not None:
            try:
                os.dup2(saved_stdout_fd, stdout_fd)
                os.dup2(saved_stderr_fd, stderr_fd)
                os.close(saved_stdout_fd)
                os.close(saved_stderr_fd)
            except Exception:
                pass
        else:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        devnull.close()


@dataclass
class ExternalResult:
    ok: bool
    method: str
    error: str = ""
    model: object = None
    tokenizer: object = None
    kind: str = "causal_lm"
    stats: Dict = None


def load_bitsandbytes(model_id: str, method: str, *, device: str, trust_remote_code: bool) -> ExternalResult:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except Exception as exc:
        return ExternalResult(False, method, f"Missing transformers/bitsandbytes support: {exc}")
    try:
        if method == "bnb_int8":
            qconf = BitsAndBytesConfig(load_in_8bit=True)
        elif method == "bnb_nf4":
            qconf = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            raise ValueError(method)
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=qconf,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        return ExternalResult(True, method, model=model, tokenizer=tok, stats={"backend": "bitsandbytes"})
    except Exception as exc:
        return ExternalResult(False, method, f"{type(exc).__name__}: {exc}")


def quantize_awq(model_id: str, calib_texts: List[str], *, method: str, device: str, trust_remote_code: bool) -> ExternalResult:
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except Exception as exc:
        return ExternalResult(False, method, f"Install autoawq: {exc}")
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
            tok.pad_token = tok.eos_token
        with suppress_output():
            model = AutoAWQForCausalLM.from_pretrained(model_id, trust_remote_code=trust_remote_code, safetensors=True)
            qconf = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
            model.quantize(tok, quant_config=qconf, calib_data=calib_texts)
        return ExternalResult(True, method, model=model.model if hasattr(model, "model") else model, tokenizer=tok, stats={"backend": "autoawq", **qconf})
    except Exception as exc:
        return ExternalResult(False, method, f"{type(exc).__name__}: {exc}")


def quantize_gptq(model_id: str, calib_texts: List[str], *, method: str, device: str, trust_remote_code: bool) -> ExternalResult:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
            tok.pad_token = tok.eos_token
        
        qconf = GPTQConfig(
            bits=4, 
            dataset=calib_texts, 
            tokenizer=tok,
            group_size=128, 
            desc_act=False,
            modules_to_not_convert=["lm_head"] # leave lm_head dense for our ARCHead combination
        )
        
        with suppress_output():
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                quantization_config=qconf, 
                trust_remote_code=trust_remote_code,
                device_map="auto"
            )
        return ExternalResult(True, method, model=model, tokenizer=tok, stats={"backend": "optimum_gptq", "bits": 4, "group_size": 128})
    except Exception as exc:
        return ExternalResult(False, method, f"Optimum GPTQ failed: {exc}")


EXTERNAL_METHODS = {"bnb_int8", "bnb_nf4", "awq_4bit", "gptq_4bit"}


def run_external_loader(method: str, model_id: str, calib_texts: List[str], *, device: str, trust_remote_code: bool) -> ExternalResult:
    if method in {"bnb_int8", "bnb_nf4"}:
        return load_bitsandbytes(model_id, method, device=device, trust_remote_code=trust_remote_code)
    if method == "awq_4bit":
        return quantize_awq(model_id, calib_texts, method=method, device=device, trust_remote_code=trust_remote_code)
    if method == "gptq_4bit":
        return quantize_gptq(model_id, calib_texts, method=method, device=device, trust_remote_code=trust_remote_code)
    raise ValueError(method)

