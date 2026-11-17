# melon-chart

멜론 Top 100 차트를 FastAPI로 가공하고 Gemini로 요약한 결과를 보여주는 독립 페이지입니다.

- 주소: https://partial02.github.io/melon-chart/
- 사이트의 다른 파일은 전혀 수정하지 않았습니다. **이 `melon-chart/` 폴더만 지우면 원상복귀됩니다.**
- 서버 없이 `data/*.json` 정적 파일만 읽으므로 누구나 볼 수 있습니다. (Gemini 키는 올라가지 않습니다)

## 데이터 갱신

API 프로젝트 폴더에서:

```
python melon_top100.py     # 최신 차트로 melon_top100.csv 갱신 (선택)
python export_static.py    # API 결과를 melon-chart/data/*.json 으로 저장
```

그다음 이 저장소에서 `melon-chart/data` 를 commit & push 하면 사이트에 반영됩니다.
