# 文献の照合（指示書 §2.5、2026-10-26）

- 対象：`paper/draft/references.bib`（35 エントリ）と、本文（`paper/draft/` の main.tex と 7 つの .tex、コメントを除く）の `\cite`。
- 確かめ方：DOI のあるもの（23 件）は Crossref の API（`/works/<DOI>`）、DOI のないもの（12 件）は Crossref の検索（`query.bibliographic`、3 件まで）。Crossref にない会議録（PMLR）は PMLR の頁の citation メタデータで確かめた。問い合わせは 1 件ずつ間をあけて合計 41 件。
- 原稿は変えていない。

---

## 1. 食い違い・確かめられないもの

食い違いはどれも書誌の細部（年・号・DOI の有無・著者の省略）で、題名・誌名・巻・頁の誤りはない。

| キー | 項目 | bib の記述 | Crossref など一次の記述 | DOI |
|---|---|---|---|---|
| noguchi2022multi | 年 | year = 2022 | Crossref の issued は 2021-11-02（オンライン公開）、published-print は 2022-01-02、巻 36・号 1-2・頁 85–99。冊子の年（2022）なら bib は正しい。どちらの年で書くかは投稿先の書式による（判断が要る） | 10.1080/01691864.2021.1993334 |
| rabinowitz2018machine | 著者 | `Rabinowitz, Neil and others`（文献 [1] の表記のまま） | PMLR v80 の頁：Neil Rabinowitz, Frank Perbet, Francis Song, Chiyuan Zhang, S. M. Ali Eslami, Matthew Botvinick。題名・頁 4218–4227・年 2018 は一致。unsrt では "and others" は "et al." と出るので誤りではないが、ほかのエントリは全著者を書いている | なし（PMLR） |
| gopnik1994theory | DOI | DOI なし | Crossref に章としてある：Gopnik, Alison; Wellman, Henry M.「The theory theory」『Mapping the Mind』Cambridge University Press、頁 257–293、1994。bib と一致。DOI を足すなら 10.1017/CBO9780511752902.011（編者は Crossref に記載なし。bib の編者 Hirschfeld・Gelman は確かめられない） | 10.1017/cbo9780511752902.011 |
| noguchi2017cognitive | 号・DOI | 号なし、DOI なし | Crossref：Adaptive Behavior 25(3), 129–146, 2017。著者・題名・巻・頁・年は一致。号 3 と DOI がない | 10.1177/1059712317711487 |
| gordon1986folk | 号・DOI | 号なし、DOI なし | Crossref：Mind & Language 1(2), 158–171, 1986。号 2 と DOI がない | 10.1111/j.1468-0017.1986.tb00324.x |
| gallese1998mirror | 号・DOI | 号なし、DOI なし | Crossref：Trends in Cognitive Sciences 2(12), 493–501, 1998（著者は Crossref では Gallese だけ載っている。Goldman は Crossref の記載にないが、論文の著者として広く知られている。ここでは確かめられないとする） | 10.1016/S1364-6613(98)01262-5 |
| rizzolatti2004mirror | 号・DOI | 号なし、DOI なし | Crossref：Annual Review of Neuroscience 27(1), 169–192, 2004（題名は Crossref では大文字 "THE MIRROR-NEURON SYSTEM"）。号 1 と DOI がない | 10.1146/annurev.neuro.27.070203.144230 |
| danjo2018spatial | 号・DOI | 号なし、DOI なし | Crossref：Science 359(6372), 213–218, 2018 | 10.1126/science.aao3898 |
| omer2018social | 号・DOI | 号なし、DOI なし | Crossref：Science 359(6372), 218–224, 2018 | 10.1126/science.aao3474 |
| baker2017rational | 号・DOI | 号なし、頁 0064、DOI なし | Crossref：Nature Human Behaviour 1(4)、論文番号 0064、2017 | 10.1038/s41562-017-0064 |
| goldman2006simulating | （一致） | Oxford University Press, New York, 2006 | Crossref の題名は副題なしの "Simulating Minds"（副題は Crossref に載っていない）。出版社・年・DOI は一致 | 10.1093/0195138929.001.0001 |
| ng2000algorithms | 全項目 | ICML 2000, pp. 663–670 | **確かめられない**（Crossref に ICML 2000 の会議録はない。検索でも該当なし） | なし |
| auchi2022understanding | 全項目 | AROB 2022（第27回）, pp. 1812–1817 | **確かめられない**（Crossref の検索で該当なし。bib のコメントのとおり DOI なし。中身も【要確認】のまま） | なし |

- 号・DOI がないもの（[A] 区分の8件）は、文献 [1] の表記をそのまま使ったため。誤りではないが、ほかの [V] のエントリと書き方がそろっていない。足すかどうかは書式の判断。

## 2. 全エントリの表

| キー | 区分（bib のコメント） | 確かめ方 | 判定 |
|---|---|---|---|
| noguchi2022superposition | [V] | Crossref（DOI） | 一致（著者4名・題名・Sci. Rep. 12・論文番号 2859・2022） |
| noguchi2017cognitive | [A] | Crossref（検索） | 一致（号 3・DOI が bib にない） |
| gopnik1994theory | [A] | Crossref（検索の2番目の候補を DOI で取得） | 一致（DOI なし、編者は確かめられない） |
| gordon1986folk | [A] | Crossref（検索） | 一致（号 2・DOI がない） |
| goldman2006simulating | [V] | Crossref（DOI） | 一致（Crossref の題名に副題なし） |
| gallese1998mirror | [A] | Crossref（検索） | 題名・誌名・巻・頁・年は一致（号 12・DOI がない。第二著者は Crossref に記載なし） |
| rizzolatti2004mirror | [A] | Crossref（検索、2回目の問い合わせで該当） | 一致（号 1・DOI がない） |
| danjo2018spatial | [A] | Crossref（検索） | 一致（号・DOI がない） |
| omer2018social | [A] | Crossref（検索） | 一致（号・DOI がない） |
| baker2009action | [V] | Crossref（DOI） | 一致 |
| baker2017rational | [A] | Crossref（検索） | 一致（号 4・DOI がない） |
| rabinowitz2018machine | [A] | PMLR の頁（Crossref になし） | 題名・巻・頁・年は一致。著者は "and others"（6名） |
| raileanu2018modeling | [V] | PMLR の頁（Crossref になし） | 一致（4名・頁 4257–4266・2018） |
| woodward1998infants | [V] | Crossref（DOI） | 一致（Crossref の著者名は Woodward のみ、bib は "Woodward, A."） |
| sommerville2005action | [V] | Crossref（DOI） | 一致 |
| repacholi1997early | [V] | Crossref（DOI） | 一致 |
| jern2017people | [V] | Crossref（DOI） | 一致 |
| wolpert2003unifying | [V] | Crossref（DOI） | 一致 |
| keysers2007integrating | [V] | Crossref（DOI） | 一致 |
| kilner2007predictive | [V] | Crossref（DOI） | 一致 |
| decety2003shared | [V] | Crossref（DOI） | 一致 |
| gordon2021simulation | [V] | Crossref（DOI） | 一致（編者は Crossref の記載になく確かめられない） |
| noguchi2022multi | [V] | Crossref（DOI） | 年に注意（オンライン 2021、冊子 2022）。ほかは一致 |
| ng2000algorithms | [V] | Crossref（検索） | 確かめられない |
| nickerson1999how | [V] | Crossref（DOI） | 一致 |
| epley2004perspective | [V] | Crossref（DOI） | 一致 |
| ross1977false | [V] | Crossref（DOI） | 一致 |
| vanboven2003social | [V] | Crossref（DOI） | 一致 |
| saxe2005against | [V] | Crossref（DOI） | 一致 |
| mitchell2006dissociable | [V] | Crossref（DOI） | 一致 |
| suzuki2012learning | [V] | Crossref（DOI） | 一致（8名） |
| tamir2010neural | [V] | Crossref（DOI） | 一致 |
| copete2016motor | [V] | Crossref（DOI） | 一致 |
| auchi2022understanding | [V] | Crossref（検索） | 確かめられない |
| geirhos2020shortcut | [V] | Crossref（DOI） | 一致（7名） |

- 大文字・小文字、引用符、ダッシュの違い（例：Crossref の "Learning to Simulate Others' Decisions" と bib の "Learning to simulate others' decisions"）は食い違いに数えていない。

## 3. \cite と bib の対応

- bib のエントリ 35、本文（コメントを除く）で引かれたキー 35。
- **本文で引かれているが bib にないキー：なし**。
- **bib にあるが本文で引かれていないキー：なし**。
- コメントの中だけにある \cite：なし。
- 引用の多いもの：noguchi2022superposition 28 か所、goldman2006simulating 4 か所。ほかは 1〜3 か所（intro・related・discussion に分布）。
