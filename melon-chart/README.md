# melon-chart

멜론 Top 100 차트를 FastAPI로 가공하고 Gemini로 요약한 결과를 보여주는 독립 페이지입니다.

- 페이지: https://partial02.github.io/melon-chart/
- 사이트의 다른 파일은 전혀 수정하지 않았습니다. **이 `melon-chart/` 폴더만 지우면 원상복귀됩니다.**

## API (로컬 FastAPI 와 같은 경로)

| 로컬 | GitHub Pages |
|---|---|
| `http://127.0.0.1:8000/songs` | https://partial02.github.io/melon-chart/songs |
| `http://127.0.0.1:8000/songs/top10` | https://partial02.github.io/melon-chart/songs/top10 |
| `http://127.0.0.1:8000/artists` | https://partial02.github.io/melon-chart/artists |
| `http://127.0.0.1:8000/insight` | https://partial02.github.io/melon-chart/insight |
| `http://127.0.0.1:8000/docs` | https://partial02.github.io/melon-chart/docs |

GitHub Pages 는 서버를 돌릴 수 없으므로, 각 경로를 폴더로 만들고 `index.html` 에 JSON 결과만 넣은 **정적 복사본**입니다.
GET 요청만 되고, 마지막으로 내보낸 시점의 데이터이며, Content-Type 은 `text/html` 입니다. (`fetch().json()`, `requests.get().json()` 은 그대로 동작)
Gemini 키는 올라가지 않습니다.

## 데이터 갱신

API 프로젝트 폴더에서:

```
python melon_top100.py     # 최신 차트로 melon_top100.csv 갱신 (선택)
python export_static.py    # 각 경로의 JSON + openapi.json + docs 다시 생성
```

그다음 이 저장소에서 `melon-chart` 를 commit & push 하면 사이트에 반영됩니다.
