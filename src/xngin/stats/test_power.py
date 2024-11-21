from xngin.stats.power import MetricSpec, MetricType, analyze_metric_power, check_power

def test_analyze_metric_power_continuous():
    metric = MetricSpec(
        metric_name="test_metric",
        metric_type=MetricType.CONTINUOUS,
        metric_baseline=100,
        metric_target=110,
        metric_stddev=20,
        metric_available_n=1000
    )
    
    result = analyze_metric_power(metric, n_arms=2)
    
    assert result.metric_name == "test_metric"
    assert result.metric_type == MetricType.CONTINUOUS
    assert result.metric_baseline == 100
    assert result.metric_target == 110
    assert result.available_n == 1000
    assert result.target_n == 128.0
    assert result.sufficient_n
    assert result.msg is not None

def test_analyze_metric_power_binary():
    metric = MetricSpec(
        metric_name="test_metric",
        metric_type=MetricType.BINARY,
        metric_baseline=0.5,
        metric_target=0.55,
        metric_available_n=1000
    )
    
    result = analyze_metric_power(metric, n_arms=2)
    
    assert result.metric_type == MetricType.BINARY
    assert result.metric_baseline == 0.5
    assert result.metric_target == 0.55
    assert isinstance(result.target_n, int)
    assert isinstance(result.sufficient_n, bool)

def test_analyze_metric_power_baseline_only():
    metric = MetricSpec(
        metric_name="test_metric",
        metric_type=MetricType.CONTINUOUS,
        metric_baseline=100,
        metric_stddev=20,
        metric_available_n=1000
    )
    
    result = analyze_metric_power(metric, n_arms=2)
    
    assert result.metric_target is None
    assert result.metric_target_possible is not None
    assert result.metric_pct_change_possible is not None
    assert result.delta is not None

def test_check_power_multiple_metrics():
    metrics = [
        MetricSpec(
            metric_name="metric1",
            metric_type=MetricType.CONTINUOUS,
            metric_baseline=100,
            metric_target=110,
            metric_stddev=20,
            metric_available_n=1000
        ),
        MetricSpec(
            metric_name="metric2",
            metric_type=MetricType.BINARY,
            metric_baseline=0.5,
            metric_target=0.55,
            metric_available_n=1000
        )
    ]
    
    results = check_power(metrics, n_arms=2)
    
    assert len(results) == 2
    assert results[0].metric_name == "metric1"
    assert results[1].metric_name == "metric2" 