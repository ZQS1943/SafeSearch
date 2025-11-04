# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
#
# SPDX-License-Identifier: CC-BY-NC-4.0
import requests
from typing import List, Union, Tuple
import json
class LocalRetriever:
    """
    Thin wrapper around Search-R1 retrieval server.
    """

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8000/retrieve",
        top_k: int = 10,
        return_scores: bool = True,
        timeout: int = 30,
    ):
        self.endpoint = endpoint
        self.top_k = top_k
        self.return_scores = return_scores
        self.timeout = timeout

    def search(
        self,
        query: str,
        top_k = None,
        return_scores = None,
    ) -> Union[List[str], List[Tuple[str, float]]]:
        """
        Args
        ----
        query : single natural-language query
        top_k : override default k
        return_scores : if True, also return retriever similarity scores

        Returns
        -------
        • list[str]                                (default)  
        • list[tuple[str, float]]                  (if return_scores=True)
        """
        k = top_k if top_k is not None else self.top_k
        rs = return_scores if return_scores is not None else self.return_scores

        payload = {
            "queries": [query],   # *** MUST be a list, field name “queries” ***
            "topk": k,            # *** arg name is “topk”, no underscore    ***
            "return_scores": rs,
        }
        # print(payload)
        resp = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        # print(data)
        # print(json.dumps(data, indent=2))
        docs = data["result"][0]               # first (and only) batch element
        final_docs = []
        for d in docs:  
            # print(d)
            key_name = "content" if "content" in d["document"] else "text" if "text" in d["document"] else "contents"

            if rs:   
                # score with 4 digits
                tmp = f"Score - [{d['score']:.4f}]; Doc - {d['document'][key_name]}"                   # each doc in the batch
            else:
                tmp = f"Doc - {d['document'][key_name]}"                   # each doc in the batch
            # print(tmp)
            # exit()
            final_docs.append(tmp)

        # default: just the passage texts
        return final_docs

if __name__ == "__main__":
    retriever = LocalRetriever()

    # Example usage
    query = " famous bridge in Venice"
    print(f"Query: {query}\n")
    results = retriever.search(query, top_k=10, return_scores=True)

    for result in results:
        print(result)
