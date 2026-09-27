# JCS&T 원고 형식 변환

대상: Universidad Nacional de La Plata의 Journal of Computer Science & Technology.
중국 ICT/CAS의 동명 JCST와 구분한다.

## 기준

- 공식 투고 지침: https://journal.info.unlp.edu.ar/JCST/about/submissions
- 공식 최신 다운로드: https://journal.info.unlp.edu.ar/public/journals/2/JCS_T_v1_12Latex_template.zip
- 다운로드 날짜: 2026-09-27. 제공 클래스의 날짜: 2026-06-09.
- 윤리 지침: https://journal.info.unlp.edu.ar/JCST/Ethics
- v1.12 템플릿의 일반 원고 한도는 15쪽이다. 인터넷의 예전 7쪽 제한 템플릿을 사용하지 않았다.

## 적용 사항

공식 jcst.cls를 수정하지 않고 사용했다. A4, 본문 10pt, 2단, 단 간격 1cm, 페이지 번호 없음이다. 제목·초록·키워드에 스페인어 번역을 추가했다. 초록은 각각 200단어 미만이고 키워드는 언어별 알파벳순 5개다. 그림은 분석값으로 생성한 300dpi PNG이며 모양으로도 국면을 구분한다. 넓은 표는 양단에 걸쳐 배치했다. 기존 11개 결과 표와 수치는 유지하면서 기물 수별 독립 IRL과 Stockfish 제한 강도 1500/2000/2500 비교를 추가했다. 참고문헌은 공식 제공 IEEEtran BibTeX 스타일로 생성했다.

저자는 사용자가 제공한 Jungbin Kim / UNIST Department of Engineering / rlawjdqls1212@unist.ac.kr / 0009-0000-4719-1228이다. 단독 저자를 교신저자로 표시했다. 저자 기여와 연구비 없음은 사용자 답변을 반영했다. 연구 결과를 기물의 실제 교환 가치로 주장하지 않았다.

## 제출 전에 남은 사항

- 최종 원고, 스페인어 번역, AI 사용 선언, 이해충돌 선언을 저자가 검토한다.
- 투고용 커버레터와 이해충돌 없는 추천 심사위원 최소 3명이 별도로 필요하다. 이 형식 변환 작업은 실제 투고를 수행하지 않았다.
- 공식 템플릿은 AI 활용을 가독성·언어 개선에 한정하는 문구를 포함하지만, 웹 윤리 지침은 연구 설계에 포함된 사용과 명확한 공개도 다룬다. 이번 연구는 코드·분석·초안 작성에도 Codex를 사용했으므로 그 사실을 숨기지 않고 원고에 명시했다. 커버레터에도 실제 사용 범위를 설명하고 편집부에 해당 정책 적용을 확인하는 것이 필요하다. 템플릿의 저자 검토 완료 문구는 실제 최종 검토 이후 확정한다.
- 코드·데이터 공개 저장소: https://github.com/rlawjdqls1212/piece-values-irl . 영구 보존용 DOI는 아직 발급하지 않았다.
- Citation box를 유지하되 DOI·권호·접수일·게재일을 만들어 넣지 않았다. 해당 정보는 편집부가 확정한다.

## 파일 및 재생성

- `output/pdf/jcst_manuscript.pdf`: JCS&T 형식 원고.
- `output/jcst_manuscript_source.zip`: main.tex, main.bbl, references.bib, jcst.cls, ieeetran.bst, stage_values.png.
- 기존 arXiv 형식 PDF와 원고는 별도로 보존했다.

프로젝트 루트에서 `make_count_section.py`, `make_elo_section.py`, `make_jcst.py` 순서로 실행해 새 결과 표와 원고를 생성한다. 모든 명령은 `.venv\Scripts\python.exe research\스크립트명.py` 형식이다. LaTeX 설치 경로는 TEX_BIN 환경변수 또는 스크립트의 TEX 상수로 설정할 수 있다.
소스 ZIP을 별도 폴더에 푼 뒤 `pdflatex main`, `bibtex main`, `pdflatex main`, `pdflatex main` 순서로 컴파일할 수도 있다.
