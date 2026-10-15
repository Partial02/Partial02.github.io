# GraphScape (CHI 2017) 논문 리뷰 · 체험 페이지

`partial02.github.io/graphscape/` 에서 열리는 정적 페이지입니다. 빌드 과정 없이 이 폴더를 저장소 루트에 그대로 두면 됩니다.

## 구조
- `index.html` : 본문(리뷰 글). 문구 수정은 여기서 합니다.
- `css/style.css` : 짙은 남색 + 비비드 팔레트와 레이아웃. 색은 맨 위 `:root` 변수에서 바꿉니다(밝은 테마는 data-theme="light").
- `js/core.js` : 공통 렌더링, graphscape 호출, 편집 연산 한국어 라벨
- `js/nav.js` : 경로 레일(진행률), 흐름도, 발표 타이머. 키보드 `[` `]` 섹션 이동, `m` 흐름도
- `js/s-*.js` : 섹션별 체험(초록, 문제, 방법론1, 실험1, 방법론2, 실험2, 응용, 마무리 s-meta). 모든 체험에 초기화 버튼(App.addReset)
- `js/data.js` : vega-datasets의 cars, flights-10k, stocks(AMZN) 압축본
- `lib/graphscape.min.js` : uwdata/graphscape 빌드 (BSD-3-Clause, UW Interactive Data Lab)

## 외부 스크립트 (버전 고정)
vega@5.25.0, vega-lite@4.17.0, d3@6.7.0 (jsDelivr). graphscape가 vega-lite 4를 전제로 해서 4.x를 씁니다.

## 수정할 곳
`index.html`의 "작성 필요" 상자 두 개(수업 주제, 내 프로젝트와의 연결)를 채워 주세요.
