/**
 * 터진 까닭을 **엔진에 상관없이** 사람이 읽을 수 있게.
 *
 * 러너 페이지들이 실패를 `err.stack` 으로 찍고 있었다. V8 은 스택의 첫 줄이
 * `Error: 무엇이 잘못됐는지` 라서 크롬에서는 멀쩡해 보인다 — **WebKit 의 스택에는
 * 메시지가 없다.** 부른 자리만 있다. 그래서 사파리로 연 사람은
 * `@http://…/device.js:95:28` 만 보게 되고, 그 순간이 정확히 설명이 가장 필요한
 * 순간이다.
 *
 * 사이트 세션이 자기 실행 블록에서 같은 것을 먼저 잡았고, 이 저장소의 러너에도
 * 열한 곳이 있었다. 크롬에서만 보면 영원히 안 드러나는 갈래다.
 *
 * 스택을 버리지는 않는다 — 메시지가 이미 들어 있으면 그대로 쓰고, 없으면 앞에 붙인다.
 */
export function why(err) {
  if (!(err instanceof Error)) return String(err);
  const head = `${err.name}: ${err.message}`;
  if (!err.stack) return head;
  return err.stack.includes(err.message) ? err.stack : `${head}\n${err.stack}`;
}
