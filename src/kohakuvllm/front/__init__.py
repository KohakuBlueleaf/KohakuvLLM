"""Our frontend for vLLM's engine driven in-process: requests prepared in worker processes
(``prep``), images encoded by our vision tower (``vision``), handed to ``AsyncLLM`` as token ids
and image embeddings (``engine``)."""
