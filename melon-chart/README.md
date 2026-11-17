# melon-chart

사이트 주인의 컴퓨터에서 실행 중인 FastAPI 서버(멜론 Top 100 + Gemini 요약)를 불러와 보여주는 독립 페이지입니다.

- 페이지: https://partial02.github.io/melon-chart/
- 사이트의 다른 파일은 전혀 수정하지 않았습니다. **이 `melon-chart/` 폴더만 지우면 원상복귀됩니다.**

## 동작 방식

- 내 컴퓨터의 `http://127.0.0.1:8000` 서버를 Cloudflare Tunnel 이 `https://xxxx.trycloudflare.com` 으로 열어 줍니다.
- 그 주소는 `api.json` 에 기록되고, 페이지는 이 주소로 API를 호출합니다. 그래서 **다른 사람 컴퓨터에서도** 내 서버의 데이터가 보입니다.
- `/melon-chart/songs`, `/songs/top10`, `/artists`, `/insight`, `/docs` 로 들어오면 실제 API의 같은 경로로 이동합니다.
- 서버를 끄면 페이지와 API 링크 모두 "서버 꺼짐" 안내가 나옵니다.

## 켜기

API 프로젝트 폴더에서 `python start_public.py` 실행
(서버 + 터널 실행 → 새 터널 주소를 `api.json` 에 써서 자동 push, 반영까지 약 1분). 끌 때는 Ctrl+C.
