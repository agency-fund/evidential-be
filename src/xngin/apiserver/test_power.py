from xngin.stats.power import analyze_metric_power


def test_metric_power_analysis():
    result = analyze_metric_power(...)
    assert isinstance(result, MetricPowerAnalysis)
    assert isinstance(result.msg, MetricPowerAnalysisMessage)
    assert result.msg.type == MetricPowerAnalysisMessageType.SUFFICIENT
