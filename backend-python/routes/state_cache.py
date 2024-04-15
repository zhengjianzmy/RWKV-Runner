from typing import Any, Dict, List, Union
from utils.log import quick_log
from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel
import gc
import copy
import global_var

router = APIRouter(include_in_schema=False)

trie = None
dtrie: Dict = {}
max_trie_len = 300
loop_start_id = 1  # to prevent preloaded prompts from being deleted
loop_del_trie_id = loop_start_id


def init():
    global trie
    try:
        import cyac

        # import mmap
        # import os
        #
        # if os.path.exists("state_cache.trie"):
        #     with open("state_cache.trie", "r") as bf:
        #         buff_object = mmap.mmap(bf.fileno(), 0, access=mmap.ACCESS_READ)
        #     trie = cyac.Trie.from_buff(buff_object, copy=False)
        # else:
        trie = cyac.Trie()
    except ModuleNotFoundError:
        print("cyac not found")


@router.post("/disable-state-cache", tags=["State Cache"])
def disable_state_cache():
    global trie, dtrie

    if global_var.get(global_var.Deploy_Mode) is True:
        raise HTTPException(status.HTTP_403_FORBIDDEN)

    trie = None
    dtrie = {}
    gc.collect()

    print("state cache disabled")
    return "success"


@router.post("/enable-state-cache", tags=["State Cache"])
def enable_state_cache():
    global trie, dtrie

    if global_var.get(global_var.Deploy_Mode) is True:
        raise HTTPException(status.HTTP_403_FORBIDDEN)

    try:
        import cyac

        trie = cyac.Trie()
        dtrie = {}
        gc.collect()

        print("state cache enabled")
        return "success"
    except ModuleNotFoundError:
        print("state cache disabled")
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "cyac not found")


class AddStateBody(BaseModel):
    prompt: str
    tokens: List[Union[str, int]]
    state: Any
    logits: Any


# @router.post("/add-state", tags=["State Cache"])
def add_state(body: AddStateBody):
    global trie, dtrie, loop_del_trie_id

    # if global_var.get(global_var.Deploy_Mode) is True:
    #     raise HTTPException(status.HTTP_403_FORBIDDEN)

    if trie is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "trie not loaded")

    import torch
    import numpy as np

    try:
        devices: List[torch.device] = []
        state: Union[Any, None] = None

        if body.state is not None:
            if type(body.state) == list and hasattr(body.state[0], "device"):  # torch
                devices = [tensor.device for tensor in body.state]
                state = [tensor.cpu() for tensor in body.state]
            elif type(body.state) == np.ndarray:  # rwkv.cpp
                state = body.state
            else:  # WebGPU
                state = body.state.back()

        id: int = trie.insert(body.prompt)
        dtrie[id] = {
            "tokens": body.tokens,
            "state": state,
            "logits": body.logits,
            "devices": devices,
        }

        if len(trie) >= max_trie_len:
            del_prompt = trie[loop_del_trie_id]
            trie.remove(del_prompt)
            dtrie[loop_del_trie_id] = None
            loop_del_trie_id = loop_del_trie_id + 1
            if loop_del_trie_id >= max_trie_len:
                loop_del_trie_id = loop_start_id

        quick_log(
            None,
            None,
            f"New Trie Id: {id}\nTrie Len: {len(trie)}\nTrie Buff Size: {trie.buff_size()}\nDtrie Buff Size Of Id: {__get_a_dtrie_buff_size(dtrie[id])}",
        )
        return "success"
    except Exception as e:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"insert failed, bad prompt.\n{e}"
        )


@router.post("/reset-state", tags=["State Cache"])
def reset_state():
    global trie, dtrie

    if global_var.get(global_var.Deploy_Mode) is True:
        raise HTTPException(status.HTTP_403_FORBIDDEN)

    if trie is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "trie not loaded")

    import cyac

    trie = cyac.Trie()
    dtrie = {}
    gc.collect()

    return "success"


class LongestPrefixStateBody(BaseModel):
    prompt: str


def __get_a_dtrie_buff_size(dtrie_v):
    # print(sys.getsizeof(dtrie_v["tokens"][0]))  # str
    # print(sys.getsizeof(dtrie_v["tokens"][0]) * len(dtrie_v["tokens"]))
    # print(dtrie_v["state"][0][0].element_size())
    # print(dtrie_v["state"][0].nelement())
    # print(len(dtrie_v["state"]))
    # print(
    #     len(dtrie_v["state"])
    #     * dtrie_v["state"][0].nelement()
    #     * dtrie_v["state"][0][0].element_size()
    # )
    # print(dtrie_v["logits"][0].element_size())
    # print(dtrie_v["logits"].nelement())
    # print(dtrie_v["logits"][0].element_size() * dtrie_v["logits"].nelement())
    return 54 * len(dtrie_v["tokens"]) + 491520 + 262144 + 28  # TODO


# @router.post("/longest-prefix-state", tags=["State Cache"])
def longest_prefix_state(body: LongestPrefixStateBody, request: Request):
    global trie

    # if global_var.get(global_var.Deploy_Mode) is True:
    #     raise HTTPException(status.HTTP_403_FORBIDDEN)

    if trie is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "trie not loaded")

    import torch
    import numpy as np

    id = -1
    try:
        for id, len in trie.prefix(body.prompt):
            pass
    except:
        pass
    if id != -1:
        prompt: str = trie[id]
        v = dtrie[id]
        devices: List[torch.device] = v["devices"]
        state: Union[Any, None] = v["state"]

        if type(state) == list and hasattr(state[0], "device"):  # torch
            state = [tensor.to(devices[i]) for i, tensor in enumerate(state)]

        quick_log(request, body, "Hit:\n" + prompt)
        return {
            "prompt": prompt,
            "tokens": v["tokens"],
            "state": state,
            "logits": v["logits"],
        }
    else:
        return {"prompt": "", "tokens": [], "state": None, "logits": None}


# @router.post("/save-state", tags=["State Cache"])
def save_state():
    global trie

    # if global_var.get(global_var.Deploy_Mode) is True:
    #     raise HTTPException(status.HTTP_403_FORBIDDEN)

    if trie is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "trie not loaded")

    # trie.save("state_cache.trie")

    return "not implemented"
