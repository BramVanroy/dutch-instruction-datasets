import heapq
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


class PriorityQueue:
    """
    A priority queue that allows updating the priority of items in the queue. We will use this
    as a means to track the time it took to query the API for a specific querier. The querier
    that took the longest will be put at the back of the queue, so that we can query the API
    with the fastest querier first at all times with every new job.

    This can be useful if you have API rate limits, because you can query the API with the fastest
    querier first, and then wait for the slower queriers to finish. This can be useful if you have multiple
    endpoints, for instance in different parts of the world where the latency is different. E.g.,
    when it is very busy in a European endpoint, it might be less busy in an American endpoint.
    """

    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._queue, (priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]

    def update(self, item, priority):
        for idx, (_, _, queued_item) in enumerate(self._queue):
            if queued_item == item:
                self._queue[idx] = (priority, self._index, item)
                heapq.heapify(self._queue)
                break


class MultiAzureQuerier:
    """
    Class for querying the Azure OpenAI API with multiple queriers, always prioritizing the fastest querier.
    See the docstring of the PriorityQueue class for more information.
    """

    def __init__(self, queriers, max_workers=6):
        self.querier_queue = PriorityQueue()
        for querier in queriers:
            self.querier_queue.push(querier, 0)
        self.max_workers = max_workers

        if self.max_workers < 2:
            raise ValueError(
                "max_workers must be at least 2. It does not make much sense to use multiple queriers if you only have one worker."
            )

    def _query_messages(self, idx_and_messages, return_full_api_output, **kwargs):
        querier = self.querier_queue.pop()
        start_time = time.perf_counter_ns()
        result = querier._query_api(idx_and_messages, return_full_api_output, **kwargs)
        end_time = time.perf_counter_ns()
        # Record the time it took to query the API for this specific querier
        self.querier_queue.push(querier, end_time - start_time)
        return result

    def query_list_of_messages(self, list_of_messages, return_full_api_output=False, return_in_order=True, **kwargs):
        with ThreadPoolExecutor(self.max_workers) as executor:
            futures = [
                executor.submit(self._query_messages, idx_and_messages, return_full_api_output, **kwargs)
                for idx_and_messages in list_of_messages
            ]

            yielder = futures if return_in_order else as_completed(futures)
            for future in yielder:
                yield future.result()
