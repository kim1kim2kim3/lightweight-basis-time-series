$ErrorActionPreference = 'Stop'

$src = (Get-Item -LiteralPath '.\양식-논문샘플(워드) (2).doc').FullName
$out = Join-Path (Get-Location) 'Ours_국내학회_수정본.doc'

if (Test-Path $out) {
    Remove-Item $out -Force
}

function Set-ParaText($doc, $index, $text) {
    $rng = $doc.Paragraphs.Item($index).Range
    $rng.End = $rng.End - 1
    $rng.Text = $text
}

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$doc = $word.Documents.Open($src, $false, $false)

try {
    $doc.SaveAs2($out, 0)

    Set-ParaText $doc 2 "Ours: 장기 시계열 예측을 위한"
    Set-ParaText $doc 3 "파라미터 효율적 구조 합성 모델"
    Set-ParaText $doc 4 "[저자명1], [저자명2]*"
    Set-ParaText $doc 5 "[소속기관명], *[공동소속기관명]"
    Set-ParaText $doc 6 "author1@domain.ac.kr, *author2@domain.ac.kr"
    Set-ParaText $doc 8 "Ours: A Parameter-Efficient Structural Synthesis"
    Set-ParaText $doc 9 "Model for Long-Term Time Series Forecasting"
    Set-ParaText $doc 10 "[Author 1], [Author 2]*"
    Set-ParaText $doc 11 "[Affiliation], *[Co-affiliation]"

    Set-ParaText $doc 15 "장기 시계열 예측에서는 horizon이 길어질수록 예측 오차뿐 아니라 모델 규모와 계산 비용도 함께 증가한다. 본 논문은 미래 구간 전체를 큰 출력층으로 직접 회귀하는 대신, 과거 시계열의 잠재 요약으로부터 구조적 basis를 생성하고 이를 조합하여 horizon 전체를 합성하는 Ours를 제안한다."
    Set-ParaText $doc 16 "ETTh1 데이터셋에서 입력 길이 96, 예측 길이 96, 192, 336, 720 조건으로 Ours를 DLinear와 PatchTST와 비교한 결과, Ours는 모든 horizon에서 11,149개의 파라미터를 유지하면서 DLinear보다 낮은 scaled MAE를 기록하였다."
    Set-ParaText $doc 17 "또한 336-step과 720-step에서는 PatchTST의 최고 scaled MAE 대비 2% 이내의 오차를 유지하면서 훨씬 작은 모델 규모를 보였다. 이 결과는 Ours가 장기 시계열 예측에서 정확도와 파라미터 효율성 사이의 실용적인 절충점을 제공함을 보여준다."

    Set-ParaText $doc 22 "장기 시계열 예측은 전력 수요, 설비 상태, 교통 흐름 등 다양한 응용에서 중요한 문제이며, 예측 horizon이 길어질수록 오차 누적과 장기 의존성 모델링의 난도가 함께 증가한다. 특히 미래 구간이 길어질수록 출력층 규모와 계산 비용까지 커지기 때문에, 정확도뿐 아니라 구조적 효율성도 함께 고려한 모델 설계가 필요하다."
    Set-ParaText $doc 23 "최근 Informer[1], Autoformer[2], FEDformer[3]와 같은 Transformer 계열 모델과 PatchTST[4], TimesNet[5], iTransformer[8] 등이 장기 예측 성능을 개선해 왔다. 반면 Are Transformers Effective for Time Series Forecasting?[6]은 단순한 linear 계열 모델도 강한 기준선이 될 수 있음을 보였으며, 이는 backbone의 복잡성만으로는 장기 예측 성능을 충분히 설명하기 어렵다는 점을 시사한다."
    Set-ParaText $doc 24 "본 논문은 이러한 문제의식 아래 Ours를 제안한다. Ours는 미래값 전체를 큰 출력층으로 직접 예측하는 대신, 과거 시계열의 잠재 요약으로부터 구조적 basis를 생성하고 이를 조합하여 horizon 전체를 합성한다. 본 연구의 초점은 모든 조건에서 최고 정확도를 달성하는 데 있지 않으며, 적은 파라미터로도 실용적인 accuracy-efficiency trade-off를 확보할 수 있는지를 검증하는 데 있다."
    Set-ParaText $doc 25 "본 논문의 기여는 다음과 같다. 첫째, 미래 구간을 구조적 basis 조합으로 생성하는 Ours를 제안한다. 둘째, horizon 증가와 무관하게 모델 파라미터 수를 일정하게 유지하는 설계를 제시한다. 셋째, ETTh1 실험을 통해 장기 예측에서의 accuracy-efficiency trade-off를 실증적으로 확인한다."

    Set-ParaText $doc 28 "Ours는 입력 다변량 시계열을 causal encoder에 통과시켜 시간 순서를 보존하는 잠재 표현을 얻고, 이로부터 미래 구간을 설명하는 구조적 basis와 계수를 예측한다. 핵심은 horizon의 각 시점을 독립적으로 예측하지 않고, horizon 전체를 하나의 구조적 조합 문제로 다루는 데 있다."
    Set-ParaText $doc 29 "본 연구에서 사용하는 basis는 추세형(trend), 주기형(seasonal), 과도형(transient) 성분으로 구성된다. trend basis는 장기 증가·감소 경향을, seasonal basis는 반복 주기를, transient basis는 단기 감쇠 패턴을 담당하며, 이들의 조합을 통해 미래 구간의 거시적 구조를 합성한다."
    Set-ParaText $doc 30 "또한 Ours는 입력 길이나 예측 길이가 커져도 대형 horizon-dependent head를 사용하지 않기 때문에, horizon 증가에 따른 파라미터 증가를 효과적으로 억제할 수 있다. 이 특성은 긴 예측 구간에서 특히 유리하며, 모델 규모를 일정하게 유지하면서도 장기 패턴을 비교적 안정적으로 표현할 수 있게 한다."
    Set-ParaText $doc 31 "요약하면 Ours는 미래를 직접 찍는 모델이라기보다, 미래를 구성할 basis를 만들고 이를 조합하는 모델에 가깝다. 이러한 설계는 장기 시계열 예측에 필요한 inductive bias를 제공하면서도, 파라미터 효율성을 함께 확보하는 데 목적이 있다."

    Set-ParaText $doc 34 "실험은 ETT(Electricity Transformer Temperature) 벤치마크의 ETTh1 데이터를 대상으로 수행하였다. ETTh1은 전력용 변압기 운용과 관련된 다변량 시계열 데이터이며, 본 실험에서는 7개 입력 변수 중 oil temperature를 의미하는 OT를 target으로 사용하였다."
    Set-ParaText $doc 35 "입력 길이는 96, 예측 길이는 96, 192, 336, 720으로 설정하였고, 비교 모델은 Ours, DLinear, PatchTST로 구성하였다. 결과는 3개 시드 평균의 scaled MAE, scaled RMSE, 그리고 trainable parameter 수로 정리하였다. 평가지표의 초점은 절대 최고 성능뿐 아니라 모델 크기 대비 예측 효율성에 두었다."

    Set-ParaText $doc 37 "표 1은 horizon별 평균 예측 오차를, 표 2는 동일 조건에서의 모델 파라미터 수를 보여준다. 두 표를 함께 보면 Ours의 성능과 모델 규모 사이의 절충 관계를 보다 명확하게 해석할 수 있다."
    Set-ParaText $doc 108 "실험 결과, PatchTST는 ETTh1에서 scaled MAE 기준 최고 정확도를 보였지만, 그 대가로 훨씬 큰 모델 크기를 요구한다. 반면 Ours는 모든 horizon에서 11,149개의 파라미터를 유지했고, 720-step에서는 DLinear보다 약 12.5배 적은 파라미터로 scaled MAE를 0.3221에서 0.2910으로 개선하였다. 또한 336-step과 720-step에서는 PatchTST 대비 scaled MAE 차이를 각각 1.40%와 1.06%로 제한하여, 2% 정확도 허용오차 내의 parameter-efficient 대안임을 확인하였다."

    Set-ParaText $doc 110 "Ⅴ. 결 론"
    Set-ParaText $doc 111 "본 논문에서는 장기 시계열 예측을 위한 파라미터 효율적 구조 합성 모델 Ours를 제안하였다. Ours는 구조적 basis 조합을 통해 horizon 전체를 합성함으로써, 예측 길이가 증가해도 모델 크기를 일정하게 유지한다. ETTh1 실험 결과는 PatchTST가 최고 정확도를 보이는 가운데, Ours가 long horizon에서 2% 정확도 허용오차 내의 훨씬 작은 대안이 될 수 있음을 보여준다. 향후에는 추가 데이터셋 검증과 latency·memory 분석을 통해 적용 범위를 더 확장할 계획이다."

    Set-ParaText $doc 114 "[1] H. Zhou et al., Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting, AAAI, 2021."
    Set-ParaText $doc 115 "[2] H. Wu et al., Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting, NeurIPS, 2021."
    Set-ParaText $doc 116 "[3] T. Zhou et al., FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting, ICML, 2022."
    Set-ParaText $doc 117 "[4] Y. Nie et al., A Time Series is Worth 64 Words: Long-term Forecasting with Transformers, ICLR, 2023."
    Set-ParaText $doc 118 "[5] H. Wu et al., TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis, ICLR, 2023."
    Set-ParaText $doc 119 "[6] A. Zeng et al., Are Transformers Effective for Time Series Forecasting?, AAAI, 2023."
    Set-ParaText $doc 120 "[7] B. Oreshkin et al., N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting, ICLR, 2020."
    Set-ParaText $doc 121 "[8] Y. Liu et al., iTransformer: Inverted Transformers Are Effective for Time Series Forecasting, ICLR, 2024."

    $doc.Save()
    Write-Output ("OUTPUT=" + $out)
    Write-Output ("PAGES=" + $doc.ComputeStatistics(2))
    Write-Output ("WORDS=" + $doc.ComputeStatistics(0))
} finally {
    $doc.Close([ref]0)
    $word.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($doc) | Out-Null
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

