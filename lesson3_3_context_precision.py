def context_precision(relevance):
    relevant_count = 0
    precision_sum = 0.0

    for rank, is_relevant in enumerate(relevance, start=1):
        if is_relevant:
            relevant_count += 1
            precision_at_k = relevant_count / rank
            precision_sum += precision_at_k

    if relevant_count == 0:
        return 0.0

    return precision_sum / relevant_count


# 1 = relevant, 0 = irrelevant

example_a = [1, 0, 0, 0, 0]
example_b = [0, 0, 0, 0, 1]
example_c = [1, 0, 1, 0, 0]
example_d = [1, 1, 1, 1, 1]

print("Example A:", context_precision(example_a))
print("Example B:", context_precision(example_b))
print("Example C:", context_precision(example_c))
print("Example D:", context_precision(example_d))

# poetry run python lesson3_3_context_precision.py