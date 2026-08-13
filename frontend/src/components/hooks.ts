import { useEffect, useRef, useState } from "react";

/** Fetch once (or re-fetch when deps change). */
export function useFetch<T>(fn: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setError(null);
    fn()
      .then((d) => alive && setData(d))
      .catch((e: Error) => alive && setError(e.message))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { data, error, loading };
}

/** Set the document title for the current page. */
export function useTitle(title: string) {
  useEffect(() => {
    document.title = title ? `${title} · gpu-optimizer` : "gpu-optimizer";
    return () => {
      document.title = "gpu-optimizer";
    };
  }, [title]);
}

/** Poll while `active`; stops automatically when it flips false. */
export function usePoll(fn: () => void | Promise<void>, ms: number, active: boolean) {
  const saved = useRef(fn);
  saved.current = fn;
  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => void saved.current(), ms);
    return () => clearInterval(id);
  }, [ms, active]);
}
