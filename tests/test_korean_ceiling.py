"""**The Korean left in `borch-ts` may shrink and may not grow.**

The rest of the repository is English: the Python library, every root document, the
configuration, the workflows, the site's own pages. What remains is `borch-ts`, and it is
under another session's hand while features land in it daily.

## Why a ceiling rather than a rule

The obvious check — "no Korean in `borch-ts`" — is the right rule and it cannot land yet:
red on arrival, it would be skipped rather than obeyed, which is worse than absent because
it teaches people to switch checks off. So the rule that *can* land today is the one that
stops the loss getting larger.

**The measurement that argued for it.** Over thirty hours `borch-ts/src` went from 40,698
Korean characters to 45,480 and `borch-ts/test` from 44,491 to 52,721, across fifteen
commits of which one was a translation. Nothing was going wrong — features were landing,
and each arrived with Korean comments, because that is what the surrounding file looks
like. Waiting was not holding position; it was losing ground at about 11% a day.

## What a green run here does **not** mean

It does not mean a directory is English. It does not mean a file is. **The ceiling is a
derivative** — it answers "did this grow" and is silent about every absolute fact, and a
green run is compatible with 40,000 Korean characters sitting exactly where they were.

This is written down because it already misled somebody. A session translated the
characters its own commit had added, ran this, saw green, and reported "vision.ts is
translated" — the check answered *did this grow* and the sentence claimed *is this
English*. The file still held 2,883 Korean characters, and the report was believed
downstream until somebody grepped.

Having a green test in front of you is what makes it easy to stop looking. To claim a
file is English, count it:

    grep -c "[가-힣]" path/to/file

## What it costs

Nothing to read and nothing to run. It asks only that **new comments in these directories
be written in English**, which is the direction the repository has already taken
everywhere else. Translating an existing block lowers the number, and lowering it is
always allowed.

## When a number moves

Lower it. The ceilings below are a record of a debt, not a budget to spend: after a
translation pass, set them to what was measured and the ratchet holds the new floor. The
failure message prints the number to write.

If a genuinely new Korean string has to go in — a case name, a fixture, something quoted
from a Korean page — raise the ceiling **in the same commit**, with the reason in the
commit message. That makes it a decision somebody made rather than a number that drifted.

## Why the failure message asks git where the number came from

A ceiling says *how much*. It never said *measured when, in whose tree, over which files*,
and both of the ways this file has misled somebody are that missing half.

**Sideways.** A translation pass lowered `borch-ts/test` to 27,201 in a tree that did not
contain another branch's new rows. Those rows carried 204 Korean characters. Both branches
were right about their own tree and the merge was over by exactly that much, so main went
red on arrival and the first guess was that the merge had reverted the translation. It had
not. A ceiling is measured at a moment, and with two branches there are two moments.

**Downward.** The tightening test below demands the ceiling follow the count down. Move a
big file out of the directory and the count falls with no translation behind it, and the
test will *insist* the new floor be locked in — printing the line to paste. A drop nobody
earned, arriving with instructions. A ratchet cannot be surprised: it compares to its own
last value, so any movement in the good direction agrees with it.

Neither is detectable from in here. Both are the same shape as the other checks that have
gone quiet in this repository — the check is right and its input is wider than the check
claims — and a check cannot measure its own boundary. So the ceiling does not try. It
makes the **red run explain itself**, which is the hour that was actually lost.

The provenance is asked of git and not written down beside the number. Written down it
would be one more fact copied by hand, going stale the first time somebody tightens the
count without touching the note — which is the other failure this repository keeps having,
and it is the one that *is* preventable: where a fact already has a home, read it there.
"""

import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
HANGUL = re.compile(r"[가-힣]")
SUFFIXES = (".ts", ".py", ".html")

# Measured 2026-08-22. Lower these after a translation pass; see the module docstring
# before raising one.
CEILINGS = {
    "borch-ts/src": 18,          # 3132 → 3133. **One character, and it is the same key as the last two rises.**
    # Two case names moved from `브라우저는거절` to `우리는거절` when the core stopped
    # granting `.double()` in name and answering in `float32` — so a place where the
    # implementations parted became a place where they agree, and the name had to say
    # so. Shorter by two characters each, and a gap ledger row's marker made up the
    # difference, which is why this is +1 rather than −4.
    #
    # The markers (`아직`, `별칭`, `파이썬`, `없음`) and these verdict words are keys
    # that `test_alias_rows.py`, `test_site.py` and the golden case names match on.
    # They move when those move and not before.
    # 3133 → 3135. The `act::` row returning to the ledger, whose marker word `아직`
    # is what `test_site.py` matches on to tell owed work from declined work.
    #
    # 3135 → 3139. **+4, and all four are markers being named rather than used.**
    # `opt::` moved from `아직` to `없음` when borch.ts gained `maximize`, `amsgrad`,
    # `centered`, `momentum` and `decoupled_weight_decay`, and the note beside the row
    # says which marker it left and which it took — a change of ledger verdict that
    # does not say what it changed from is the kind this repository keeps finding.
    # The twelve case names that came with it are ASCII and cost nothing here.
    #
    # 3139 → 3141. **+2, the same kind again**: the `act::` row's cause was rewritten
    # (no second shader is needed — `copyFrom` is the write-back and borch.ts has it),
    # and the note says the marker `아직` was right while the reason under it was not.
    # Naming the marker costs two characters and is the whole point of the note.
    # 3141 → 3143. One golden case name, `배율` — `scaled_dot_product_attention`'s
    # `scale`, which was accepted and dropped. Names are keys: the Python and
    # TypeScript tables have to agree on the string or the row reconciles against
    # nothing.
    # 3143 → 3145. One case name, `표` — `nonzero`'s table form beside its new
    # tuple form. Both had to be asked: `torch.nonzero` is unreadable to `inspect`,
    # so the argument had never been compared on either axis.
    # 3145 → 3148. One case name, `메서드` — the method spelling of the five math
    # binaries beside the function spelling that had been asked for months. Names
    # are keys, so both tables have to say the same string or the row reconciles
    # against nothing.
    # 3148 → 3155. Two case names, `텐서 지수` and `수 지수` — `pow` with a tensor
    # exponent beside `pow` with a number. The two are different kernels with
    # different backwards, so one name could not stand for both.
    # 3155 → 3157. `quantile(0.3, …)` at the four interpolation rules the default
    # hides — the tag is `접힘` and the four new names reuse it.
    # 3157 → 3161. Two case names, `2층` and `마스크` — the transformer stack and
    # the causal mask, the last five names on the name axis.
    # 3161 → 3172. `run.py` 의 unasked 원장 한 줄 — `pad::거절::` 넷과 그 사유.
    # 3186 → 3199. `fft::stft align_to_window(center 이면 거절)` 하나 — 값이 아니라
    # 거절이 그 인자의 전부라 케이스 이름이 그렇게 길다.
    # 3180 → 3186. `misc::층::Upsampling*(크기)` 둘 — 이름이 키라 번역 대상이 아니다.
    # 3172 → 3180. `cases.ts` 에 `loss::가장자리::` 넷 — torch 의 폐기된
    # `size_average`/`reduce` 를 borch.ts 에서 실제로 접는 유일한 케이스들이다.
    # 이름은 키라서 번역 대상이 아니다.
    # 케이스 이름은 두 언어가 같은 문자열을 써야 대조되므로 원장 항목도 같은
    # 접두사를 그대로 든다.
    # 3186 → 3192. `vision::Crop(패딩1, edge|reflect|symmetric)` 셋의 `패딩` 이다.
    # 케이스 이름은 파이썬 쪽과 **글자까지 같아야** 하는 키라, 이쪽만 영어로 쓰면
    # 두 표가 서로 다른 케이스를 가리키게 된다. 번역 대상이 아니라 대조 대상이다.
    # 3205 → 3211. 원장의 `seq::…/batch_first` 셋 — 접두사가 케이스 이름의 앞부분이라
    # 한국어가 섞인다. 키의 일부이지 산문이 아니다.
    # 3192 → 3205. `fft::stft align_to_window(center 이면 거절)` 하나 — 값이 아니라
    # 거절이 그 인자의 전부라 이름이 그렇게 길다.
    # 3211 → 3234. `pool::` 열한 건의 이름 — `테두리 채움`, `가장자리 빼기`,
    # `나눗수 지정`, `올림` 셋. 전부 키이고, 그 인자들이 **파이썬 쪽에서 그렇게 불린
    # 채로 얼어 있다.** 이쪽만 영어로 쓰면 두 표가 서로 다른 케이스를 가리키고, 러너는
    # 그것을 "TS 에 없다"로 읽는다 — 진짜 구멍과 **같은 문구**라 원인이 안 보인다.
    # 새로 쓴 산문 주석은 전부 영어다. 이 증가분은 산문이 아니라 이름이다.
    # 3234 → 3236. `act::nn.…(inplace)/같은 객체` 여섯 건의 꼬리다. 템플릿 문자열
    # 하나라 넉 자만 늘었고, 같은 커밋에서 지운 `act::` 원장 항목의 두 자가 상쇄해 +2 다.
    # 이것도 키다 — 정체성을 묻는 쪽과 값을 묻는 쪽을 가르는 것이 저 꼬리이고,
    # 이쪽만 영어로 쓰면 여섯 쌍이 전부 짝을 잃는다.
    # 3236 → 3264. `misc::` 열한 건의 이름이다 — `층::`, `표가 줄었다`,
    # `색인 안 된 행`, `새 표와 준 표`, `repr::…(전부)`. 파이썬 쪽에서 그렇게 얼어 있는
    # 키이고, 이쪽만 영어로 쓰면 두 표가 서로 다른 케이스를 가리킨다.
    # 3264 → 3272. `seq::…/batch_first 는 같은 답을 돌려놓는다` 셋의 꼬리다. 키이고,
    # `run.py` 원장 항목이 지워지며 줄어든 두 자가 상쇄해 +8 이다.
    # 3272 → 3356. `v2::` 열아홉과 `unpool::` 스물의 이름이다 — `적응형softmax::`,
    # `층::repr::`, `분수::`, `이름이 둘인 같은 계산`, `표본 없이(모양과 범위)` 따위.
    # 전부 키이고, 이쪽만 영어로 쓰면 두 표가 서로 다른 케이스를 가리킨다.
    # 3356 → 3369. `dataset::IDX labels(short by two)=거절` 하나와 그 답 문자열
    # `거절|문구=`. 케이스 이름이자 **답 자체**라 — 양쪽이 같은 글자를 내놔야 대조가
    # 된다. 번역 대상이 아니라 비교 대상이다.
    # 3369 → 3370. **One character**, and it is a case name rather than prose: the
    # forty layer reprs ported to this side are written with one template literal
    # instead of forty `out.set` lines, so the shared prefix `unpool::층::repr::`
    # carries a single 층. Written out longhand it would have been forty of them.
    # 3370 → 3373. **Three characters**, and they are the ledger's own vocabulary: the
    # seven top-level losses get a row, and a row's reason has to open with one of
    # `아직` / `없음` / `별칭` / `파이썬` for `test_site.py` to split the remainder. This
    # one is 파이썬.
    # 3373 → 3375. Two more, and the same kind: v2's tv_tensor dispatch gets a row and
    # its reason opens with 없음.
    # 3375 → 3387. Twelve, and all of them **case names**, which is the allowed kind:
    # `자리::` for the group asking torch's argument seats positionally, and
    # `(두 행)` / `(빈 행)` for the two that give `multilabel_margin` a second row.
    # The names have to be the same string on both sides — the golden is keyed by
    # them — so the Python side's name is what this side must write.
    # 3387 → 3398. Eleven, and case names again: `where` asked with a mask narrower
    # than the values, and with a branch narrower than the mask. Every `where` case
    # before them handed all three the same shape, which is the one arrangement that
    # cannot tell broadcasting from three buffers read at the same offset.
    # 3398 → 3404. Six, and case names again: the maximum's `자리::` group, which asks
    # what its padding and `ceil_mode` compute rather than how the layer prints them.
    # 3404 → 3419. Fifteen, case names once more: `Upsample` at a **fractional** scale
    # factor, in three arrangements. Every `Upsample` case before them used 2, and a
    # whole factor is the one number that cannot tell flooring from not flooring, nor
    # `recompute_scale_factor` from its absence — at 1.5 the first gave a shape with a
    # `.5` in it and the second is a 10% difference in the values.
    # 3419 → 3421. Two, and the ledger's own vocabulary: a new `v2f::` row, whose reason
    # has to open with one of `아직` / `없음` / `별칭` / `파이썬` for `test_site.py` to
    # split the remainder. This one is 없음.
    # 3421 → 3441. Twenty, and case names plus the three verdicts they return: the
    # `낱말::` group asks each of `foreach`, `fused`, `capturable`, `differentiable` of
    # `SGD` and `Adam`, and answers `거절` / `받고 값이 같다` / `받는데 값이 다르다`.
    # **The verdict strings cannot be English here.** They are the frozen values in
    # `golden.json`, written by `tests/cases.py`, and the two sides have to spell them
    # identically or every one of the eight reads as a divergence.
    # 3441 → 3486. Forty-five, case names and two verdicts: the `linalg` group's six
    # closed seats — `matrix_norm(dim)`, `matrix_norm(keepdim)`, `matrix_rank(tol)`,
    # `pinv(rcond)`, four `lstsq` rows and three refusals — plus `둘 다 멈춘다` and
    # `여기선 통과했다`, which are frozen values in `golden.json` and have to be spelled
    # the same on both sides or the case reads as a divergence.
    # 3486 → 3523. Thirty-seven, case names again: `embedding` 의 여섯 (`표가
    # 짧아진다`, `안 본 줄은 그대로`, `내놓는 값`, `padding_idx 없이`) plus the two
    # `우리는거절` rows, and the two optimizer rows the Python table names. Every one
    # of these is a key in `golden.json` and has to be spelled identically on both
    # sides or the case reads as a divergence.
    # 3523 → 3600. Seventy-seven, case names once more: the eight `inplace` rows
    # (`같은 객체`, `부른 쪽 텐서`, `기본은 그대로 둔다`) and the seven `CyclicLR` ones
    # (`momentum 자취`, `momentum 이 값을 바꾼다`, `scale_mode 만으로는 안 바뀐다` …).
    # Each is a key in `golden.json` and has to be spelled identically on both sides
    # or the case reads as a divergence — and `같은 객체=True` had to take **Python's**
    # spelling of the boolean for the same reason, `${true}` giving `true`.
    # 3600 → 3627. Twenty-seven, case names: the five `(inplace)/같은 객체` rows, the
    # three `=둘 다 거절` ones and `Flatten기본`. Each is a key in `golden.json` and has
    # to be spelled the same on both sides or the case reads as a divergence.
    # 3627 → 3734. A hundred and seven, case names: the nineteen `container::` rows
    # that walk the module tree (`층 이름`, `뿌리는 빈 이름`, `자기를 돌려준다`,
    # `점 찍힌 이름`, `이름을 나중에`, `add_param_group/스텝` …). Each is a key in
    # `golden.json` and has to be spelled identically on both sides or the case reads
    # as a divergence.
    # 3734 → 3765. Thirty-one, case names: the five in-place rows the generated
    # forwarders opened (`heaviside_(values 라는 이름)`) and the six `scatter` ones
    # `reduce` opened (`제자리::scatter_(reduce=add)`, `거절::scatter(reduce=sum)`,
    # `거절::scatter(reduce) 의 기울기`). Each is a key in `golden.json`, and the two
    # refusals also **return** a fragment string that both sides spell by hand — so a
    # divergence here reads as a divergence in the answer, not in the name.
    # 3765 → 3873. A hundred and eight, case names and one verdict: the nine
    # `inplace::기울기::` rows that `backward(inputs=…)` and `retainGrad()` opened
    # (`중간 노드도 채운다`, `안 부른 잎을 안 건드린다`, `inputs 밖에서도 남는다`,
    # `거절::빈 inputs`), plus `있다`/`없다`, which is a **frozen value** in
    # `golden.json` rather than a name and has to be spelled the same on both sides.
    # 3873 → 3891. Eighteen: the two `container::BatchNorm(device|dtype)=우리는거절`
    # names and the `기대대로` / `뜻밖의 성공` verdicts they return, which are frozen
    # values in `golden.json` and have to be spelled identically on both sides.
    # 3891 → 3900. Nine, one case name: `modfn::모양::squeeze(길이가 1 이 아닌 축)`,
    # a key in `golden.json` and so spelled the same on both sides.
    # 3900 → 3911. Eleven, two case names: `제자리::transpose_(1, 2) 는 3차원에서`
    # and `제자리::squeeze_(0, 2)`. Both are keys in `golden.json`.
    # 3911 → 3918. Seven, in the two `seq::grad::TransformerEncoderLayer/` case
    # names — `입력` and `파라미터 합`. Both are keys in `golden.json`.
    # 3918 → 3923. Five, one case name: `batch::lu_solve(한쪽만 교환)`, a key in
    # `golden.json`.
    # 3923 → 3938. Fifteen, in four `kron` case names — `kron(2차원)`,
    # `kron(2차원 × 1차원)`, `kron(직사각)` and `kron(2차원)의 기울기`. All four are
    # keys in `golden.json` and so are spelled the same on both sides.
    # 3938 → 4053. A hundred and fifteen, in `lstsq`'s eight new batched case names
    # (`batch::lstsq(행렬 우변)`, `(우변 하나를 늘린다)`, `(벡터 우변은 안 늘어난다)`,
    # `(잘림은 행렬마다 본다)`, `(둘 다 잘리면)`, `(하나짜리 배치도 잘린다)` and two
    # more) plus the verdict strings they return — all of them keys or values in
    # `golden.json`, so both sides spell them the same — and the reason written beside
    # the `linalg::` row in `run.py`.
    # 4053 → 4063. Ten, in `lu_solve`'s two new batched case names
    # (`batch::lu_solve(adjoint, 한쪽만 교환)` and `(left=False, 한쪽만 교환)`) — keys in
    # `golden.json`, so both sides spell them the same. The other three new names are
    # `adjoint` and `left=False`, which are English.
    # 4063 → 4065. Two, in `tensorsolve(dims 없이, 2×3×2×3)` — a key in `golden.json`.
    # The six `dims=` names beside it are English.
    # 4065 → 4089. Twenty-four, in the two `weight::` case names that are not pure
    # ASCII — `mse_loss 의 기울기` and `모양이 다르면 거절` — plus the verdict strings
    # `문구대로` and `안 던졌다` that the second one returns. All are keys or values in
    # `golden.json`, so both sides spell them the same.
    # 4089 → 4119. Thirty, in `embedding`'s new case names — the two `없이` suffixes,
    # `scale_grad_by_freq 는 값을 안 건드린다`, the two `=우리는거절` on the bag — and the
    # verdict strings those refusals return. All are keys or values in `golden.json`,
    # so both sides spell them the same.
    # 4119 → 4140. Twenty-one, in `interpolate`'s four gradient case names
    # (`… 의 기울기`) — keys in `golden.json` — and the reason written beside the
    # `fname::` row in `run.py` for the mode a compile-time union cannot refuse.
    # 4140 → 4170. Thirty, in `unfold`/`fold`'s new case names — the four
    # `배치 없이` suffixes, `grad::unfold(배치 없이)`, `fold(배치 없이)` and the two
    # `=거절 문구` — plus the `안 던졌다` verdict. All are keys or values in
    # `golden.json`, so both sides spell them the same.
    # 4170 → 4191. Twenty-one, in `antialias`'s two gradient case names
    # (`… 의 기울기`) and `interpolate(nearest 에 antialias)=둘 다 거절` with the two
    # verdict strings it returns. All are keys or values in `golden.json`.
    # 4191 → 4193. Two, in `grid_sample(bicubic, 반 칸)` — a key in `golden.json`.
    # The other twelve `bicubic` names are English.
    # 4193 → 4230. Thirty-seven, in `InstanceNorm`'s three ported case names
    # (`(기본)/state_dict 열쇠`, `(affine)/…`, `(추적)/…`) and the `container::` reason
    # rewritten in `run.py` — that row is a Korean sentence and replacing it swapped
    # one Korean string for a longer one rather than adding a new kind.
    # 4230 → 4260. Thirty, in `ldl_factor`'s new case names — the three fixtures'
    # tags (`2x2 블록`, `교환`, `6x6 열 교환`), `(특이)` and `(영행렬)`, and the
    # `=둘 다 거절` with the two verdict strings it returns. All keys or values in
    # `golden.json`.
    # 4260 → 4283. Twenty-three, in the recurrent flags' case names — the labels
    # `양방향`, `2층`, `2층양방향`, `편향없음` and `dropout 은 평가에서 항등`, the parts
    # `출력`/`상태`/`셀`, `state_dict 열쇠` and the `(거절 없음)` a refusal case returns
    # when nothing was refused. Every one is a key or a value in `golden.json`, so
    # both sides have to spell it the same; the `proj_size` and `relu` labels beside
    # them are ASCII and add nothing.
    # 4283 → 4298. Fifteen, in `BatchNorm(추적없음)`'s four case names — the label
    # itself, `state_dict 열쇠` and `running_mean 은 None`. Keys in `golden.json`, so
    # both sides spell them the same; the `BatchNorm1d(N,C,L)` rows beside them are
    # ASCII apart from the `열쇠` they do not carry.
    # 4298 → 4312. Fourteen, in the top-level recurrent flags' case names — `양방향`,
    # `2층양방향`, `2층 dropout, 평가` and `proj_size 와 양방향`. Keys in `golden.json`,
    # so both sides spell them the same.
    # 4312 → 4333. Twenty-one, in the attention flags' case names — `둘 다`,
    # `출력`, `가중치`, `state_dict 열쇠` and the two `마스크` rows. Keys in
    # `golden.json`; `add_bias_kv`, `add_zero_attn` and `kdim, vdim` are ASCII.
    # 4333 → 4362. Twenty-nine, in `interpolate`'s rank case names — `3차원`, `5차원`,
    # `… 의 기울기` and the `=둘 다 거절` rows with the `둘 다 멈춘다` verdict they
    # return. All keys or values in `golden.json`, so both sides spell them the same.
    # 4362 → 4376. Fourteen, in the six `축마다 다른 배율` case names — one scale per
    # axis. Keys in `golden.json`.
    # 4376 → 4430. Fifty-four, in the strong-Wolfe case names — `진짜 기울기`,
    # `이력이 밀려난다`, `평가 예산이 짧다`, `없는 line_search_fn`, `얽힌 이차형식`,
    # `(3변수)`, `처음이 모자란다` — and the `(거절 없음)` a refusal case returns when
    # nothing was refused. All keys or values in `golden.json`.
    # 4430 → 4456. Twenty-six, in the pooling dilation case names — `자리::` on the
    # eleven `max_pool` rows, `grad::` on one, and the `avg_pool2d(dilation)=둘 다 거절`
    # with the `둘 다 멈춘다` verdict it returns. Keys or values in `golden.json`.
    # 4456 → 4475. Nineteen, in the `torch.norm` case names — `dim 뒤집기`, `배치`,
    # the `F.lp_pool*d(올림)` rows' `올림`, and the widened `linalg::` skip reason in
    # `run.py`. Keys in `golden.json` except the last, which is a sentence.
    # 4475 → 4499. Twenty-four, in the with-indices window case names — `자리 내놓기`
    # on twelve `max_pool2d`/`MaxPool2d` rows, `셋 다` on three of them, and
    # `창이 있는 자리` on the unpool row. Keys in `golden.json`, so both sides spell
    # them the same.
    # 4499 → 4503. Four, in the `index::` skip reason — `표기`, which names what the
    # eleven `걸음::` cases are about (`x[a:b:step]` is Python's notation and borch.ts
    # has no `[]`). A reason in `run.py`, not a case name.
    # 4503 → 4587. Eighty-four, in the `autograd.grad` case names — `기울기::` on
    # fifteen rows, `입력 둘`, `출력 둘`, `중간 텐서`, `안 쓰인 입력`, `씨앗 없는 벡터
    # 출력`, `모양이 틀린 grad_outputs`, `.grad 를 안 건드린다`, `쌓는다` — and the
    # `있다`/`None` verdict strings two of them return. Keys or values in
    # `golden.json`, so both sides spell them the same.
    # 4587 → 4601. Fourteen, in the `top::` skip reason — `난수::generator::`,
    # `살펴보기::layout::`, `살펴보기::형식::`, which name the case groups the row
    # accounts for. A reason in `run.py`, not case names.
    # 4601 → 4628. Twenty-seven: `norm::nn.LayerNorm… 는 F 와 같다` on two rows and
    # `0 텐서=` in the two `zero_grad` verdicts (keys and values in `golden.json`),
    # plus the `norm::`/`opt::`/`container::` skip reasons in `run.py` naming what
    # each row is about.
    # 4628 → 4690. Sixty-two: the `opt::SequentialLR/ChainedScheduler(…)=거절` and
    # `(last_epoch=N)/자취` names, the two `fft::istft(…)` ones, and the `문구대로`/
    # `안 던졌다` verdicts they return. Keys and values in `golden.json`, so both
    # sides spell them the same.
    # 4690 → 4738. Forty-eight: the eight `살펴보기::짝::… 의 자리는 int64` names, the
    # two `nn.RMSNorm/F.rms_norm(eps 를 크게)` ones, `isclose(equal_nan 없이)`, and the
    # `top::`/`act::` skip reasons in `run.py` naming what each row is about. Keys in
    # `golden.json`, so both sides spell them the same.
    # 4738 → 4771. Thirty-three: the `special::` skip reason in `run.py`, which names
    # the three of that namespace's thirty-two cases borch.ts does not ask — two are
    # `out=` (declined everywhere over there) and one hands `xlogy` a bare scalar
    # where TS types a `Tensor`. It is Korean because every other reason in that
    # ledger is, and a row in a different language reads as a different kind of row.
    # Nothing was added to `cases.ts`: the twenty-nine names ported are ASCII.
    # 4771 → 4810. Thirty-nine: the `video::` skip reason in `run.py`. It is the one
    # row in that ledger about a **type** rather than a backlog — borch.ts's `Image` is
    # `{data, height, width, channels}` and has no axis a frame could go in — and it is
    # Korean because every other reason there is, and a row in a different language
    # reads as a different kind of row. `cases.ts` did not move: the thirty video case
    # names are the core's and are not asked over there.
    # 4810 → 4824. Fourteen: the case name `linear_cross_entropy(기본이 -100 을
    # 건너뛴다)` in `cases.ts`, plus two in the `loss::` ledger row. **Names are keys** —
    # the Python and TypeScript tables have to spell the string identically or the row
    # reconciles against nothing, so a Korean name on one side is a Korean name on
    # both. That case is the one that catches `ignoreIndex=null` being passed through
    # rather than mapped to -100, which no other target can see.
    # 4824 → 4857. Thirty-three: the `special::` skip reason in `run.py`, rewritten
    # when that row went from 3 to 75. It is the one row in that ledger that changed
    # marker from `파이썬` to `아직` — from *a Python affordance* to *work owed on the
    # borch.ts side* — and the marker words are the keys `test_site.py` matches on to
    # split the README's remainder, so the change had to be made in the reason's own
    # language rather than beside it.
    "borch-ts/test": 4857,
}


def _countable(folder):
    """The files this rule counts. `_provenance` measures the same set in an old tree."""
    return [path for path in sorted((ROOT / folder).rglob("*"))
            if path.is_file() and path.suffix in SUFFIXES
            and "dist" not in path.parts and "node_modules" not in path.parts]


def _count(folder):
    total, per_file = 0, {}
    for path in _countable(folder):
        found = len(HANGUL.findall(path.read_text(errors="ignore")))
        if found:
            per_file[str(path.relative_to(ROOT))] = found
            total += found
    return total, per_file


def _git(*args):
    """git, or None when it cannot answer. A message is never worth failing over."""
    try:
        done = subprocess.run(("git", *args), cwd=ROOT, capture_output=True,
                              text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def _ceiling_commit(folder):
    """The commit that last wrote this folder's number, and the date, from `git blame`.

    Asked rather than recorded: a hash written into the table above would be correct on
    the day it was pasted and silently wrong from the first tightening that forgot it.
    """
    here = pathlib.Path(__file__)
    for number, line in enumerate(here.read_text().splitlines(), 1):
        if line.strip().startswith(f'"{folder}":'):
            blamed = _git("blame", "--porcelain", f"-L{number},{number}", "--",
                          str(here.relative_to(ROOT)))
            if not blamed:
                return None
            sha = blamed.split()[0][:12]
            return sha, (_git("log", "-1", "--format=%ad", "--date=short", sha) or "?")
    return None


def _files_at(sha, folder):
    """How many countable files that folder held in that commit's tree."""
    listing = _git("ls-tree", "-r", "--name-only", sha, "--", folder)
    if listing is None:
        return None
    return sum(1 for name in listing.splitlines() if name.endswith(SUFFIXES))


def _provenance(folder):
    """Where this number came from, and what has moved under it since.

    Two sentences, and both of them are about the same missing half: a ceiling is a
    measurement of one tree at one moment, and it is being compared against another.
    """
    found = _ceiling_commit(folder)
    if not found:
        return ""
    sha, date = found
    now_files = len(_countable(folder))
    said = [f"this ceiling was last written in {sha} ({date})"]

    merges = _git("rev-list", "--count", "--merges", f"{sha}..HEAD")
    if merges and merges != "0":
        said.append(
            f"{merges} merge(s) have landed since — a merge can put two numbers together "
            "that were each correct in their own branch, which reads exactly like growth")

    then = _files_at(sha, folder)
    if then is not None and then != now_files:
        way = "left" if then > now_files else "arrived"
        said.append(
            f"the directory held {then} countable files then and {now_files} now, so "
            f"files have {way} — a count that moved with them did not move by translation")
    return "\n  ".join(said)


def test_the_korean_left_in_borch_ts_does_not_grow():
    """The ceiling, per directory, with the worst files named when it is breached."""
    over = []
    for folder, ceiling in CEILINGS.items():
        total, per_file = _count(folder)
        if total > ceiling:
            worst = sorted(per_file.items(), key=lambda kv: -kv[1])[:5]
            where = _provenance(folder)
            over.append(
                f"{folder}: {total} Korean characters against a ceiling of {ceiling} "
                f"(+{total - ceiling})\n    "
                + "\n    ".join(f"{n}  {c}" for n, c in worst)
                + (f"\n  {where}" if where else ""))
    assert not over, (
        "Korean grew in a directory that is being translated:\n  " + "\n  ".join(over)
        + "\n\n  New comments in borch-ts go in English — everything else in this "
          "repository already does.\n  Raising a ceiling is allowed when a Korean string "
          "genuinely has to go in (a case\n  name, a quoted fixture); do it in the same "
          "commit and say why.")


def test_the_ceilings_name_directories_that_exist():
    """A ceiling over a directory that moved is a budget nobody is spending.

    It would sit at zero, pass forever, and read as a directory under control.
    """
    missing = [folder for folder in CEILINGS if not (ROOT / folder).is_dir()]
    assert not missing, (
        f"these ceilings name directories that are not there: {missing}. The code moved — "
        "point the ceiling at where it went, or drop the row if the Korean is gone.")


def test_a_ceiling_that_is_far_too_high_is_tightened():
    """**A ratchet nobody tightens is a ratchet that stopped working.**

    After a translation pass the count drops, the ceiling stays where it was, and the
    headroom left behind quietly permits new Korean back up to the old number — the pass
    is undone over the following weeks and every commit doing it is green.

    So it fails, in the same commit that earned the drop, and prints the line to paste.
    Tightening is copying a number, not taking a measurement.
    """
    slack = {}
    for folder, ceiling in CEILINGS.items():
        total, _ = _count(folder)
        if total and ceiling - total > ceiling * 0.1:
            slack[folder] = (total, ceiling, _provenance(folder))
    assert not slack, (
        "these ceilings are more than 10% above what is actually there — tighten them:\n  "
        + "\n  ".join(f'"{f}": {t},   # was {c}' + (f"\n  {w}" if w else "")
                      for f, (t, c, w) in slack.items())
        + "\n\n  Paste the number only where the drop was earned. This test cannot tell "
          "translation\n  from a file that left the directory, and it asks for the same "
          "line either way.")


def test_the_failure_message_can_say_where_the_number_came_from():
    """**The provenance runs only when something is red, so it is exercised here.**

    A path that runs only on failure rots without anybody noticing, and it is needed on
    exactly the day nobody wants a second problem. This calls it on a green tree.

    It is allowed to say nothing — outside a git checkout there is nothing to ask — but
    where git answers at all, the sentence has to name the commit that wrote the number.
    """
    if _git("rev-parse", "--git-dir") is None:
        return
    folder = next(iter(CEILINGS))
    said = _provenance(folder)
    assert said, (
        f"git is here and the provenance for {folder} came out empty — the blame lookup "
        "is keyed on the line that starts with the folder name, so it breaks when the "
        "table is reformatted.")
    assert "last written in" in said, said


# Measured floors, well under today's counts. They exist to catch a sweep that stopped
# finding files, not to track the directory's size.
FLOORS = {"borch-ts/src": 15, "borch-ts/test": 20}


def test_the_sweep_still_finds_files_to_count():
    """**A ceiling over an empty sweep is green forever, and reads as a clean directory.**

    `_count` walks `rglob("*")` and filters on suffix and on path parts. Narrow that
    filter by accident — a suffix dropped, a `dist` guard that starts matching real
    paths, a directory renamed — and it counts nothing. Nothing is under every ceiling,
    so the ratchet passes, the tightening test skips on `if total`, and both report
    exactly what they report when a translation pass has finished.

    There is no residue to find afterwards. A file that is never visited cannot be
    counted as unvisited, and the summary is small and healthy-looking either way.

    So the floor is asserted separately, on the **file count** rather than the character
    count, because the character count is supposed to fall to zero and the file count is
    not. Another session hit this exact shape today: a filter keyed on a field that does
    not exist reported `agree 0 / differ 0 / unreadable 0` for a namespace of 144 layers,
    and it was found by the row being too clean rather than by anything failing.
    """
    thin = {}
    for folder, floor in FLOORS.items():
        found = len(_countable(folder))
        if found < floor:
            thin[folder] = (found, floor)
    assert not thin, (
        "the sweep found fewer files than it should — it is measuring less than it "
        "claims:\n  "
        + "\n  ".join(f"{f}: {n} files against a floor of {fl}" for f, (n, fl) in thin.items())
        + "\n\n  A ceiling over a sweep that found nothing passes forever. Check SUFFIXES, "
          "the\n  `dist`/`node_modules` guards, and whether the directory moved.")


def test_every_ceiling_has_a_floor():
    """A ceiling added without a floor is the case above, waiting.

    The floor is cheap and its absence is invisible, which is the combination that means
    it will be forgotten unless something asks.
    """
    missing = [folder for folder in CEILINGS if folder not in FLOORS]
    assert not missing, (
        f"these ceilings have no floor beneath them: {missing}. Measure the file count "
        "and\n  write it into FLOORS well under what is there — it catches a sweep that "
        "stops\n  finding files, which no ceiling can.")
