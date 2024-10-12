from collections import defaultdict

def save_metric(metric_name, value):
    if metric_name in save_metric.metric_dict[save_metric.epoch]:
        save_metric.metric_dict[save_metric.epoch][metric_name].append(value)
    else:
        save_metric.metric_dict[save_metric.epoch][metric_name] = [value]

save_metric.metric_dict = defaultdict(dict)
save_metric.epoch = 1
