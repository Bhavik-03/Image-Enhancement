import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useMemo, useRef, useState } from "react";
import CompareSlider from "./components/CompareSlider";
import ModelPicker from "./components/ModelPicker";
import ProgressRing from "./components/ProgressRing";
import UploadZone from "./components/UploadZone";
import VideoCompare from "./components/VideoCompare";
import {
  enhancePhoto,
  listModels,
  pollVideoJob,
  startVideoJob,
  videoResultUrl,
  type VideoJobStatus,
} from "./lib/api";

type Kind = "image" | "video";
type Stage = "idle" | "ready" | "processing" | "done" | "error";

export default function App() {
  const [models, setModels] = useState<string[]>([]);
  const [defaultModel, setDefaultModel] = useState("");
  const [model, setModel] = useState("");

  const [file, setFile] = useState<File | null>(null);
  const [kind, setKind] = useState<Kind | null>(null);
  const [stage, setStage] = useState<Stage>("idle");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [videoStatus, setVideoStatus] = useState<VideoJobStatus | null>(null);

  const sourceUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);
  const resultUrlRef = useRef<string | null>(null);

  useEffect(() => {
    listModels()
      .then((res) => {
        setModels(res.models);
        setDefaultModel(res.default);
        setModel(res.default);
      })
      .catch(() => setErrorMsg("Could not reach the backend. Is `uvicorn api:app` running?"));
  }, []);

  useEffect(() => {
    return () => {
      if (sourceUrl) URL.revokeObjectURL(sourceUrl);
    };
  }, [sourceUrl]);

  function reset() {
    setFile(null);
    setKind(null);
    setStage("idle");
    setErrorMsg(null);
    setVideoStatus(null);
    if (resultUrlRef.current && resultUrlRef.current.startsWith("blob:")) {
      URL.revokeObjectURL(resultUrlRef.current);
    }
    resultUrlRef.current = null;
    setResultUrl(null);
  }

  function handleFile(f: File, k: Kind) {
    reset();
    setFile(f);
    setKind(k);
    setStage("ready");
  }

  async function runEnhance() {
    if (!file || !kind) return;
    setStage("processing");
    setErrorMsg(null);
    try {
      if (kind === "image") {
        const blob = await enhancePhoto(file, model);
        const url = URL.createObjectURL(blob);
        resultUrlRef.current = url;
        setResultUrl(url);
      } else {
        const jobId = await startVideoJob(file, model, 0.3);
        await pollVideoJob(jobId, setVideoStatus);
        const url = videoResultUrl(jobId);
        resultUrlRef.current = url;
        setResultUrl(url);
      }
      setStage("done");
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Something went wrong.");
      setStage("error");
    }
  }

  return (
    <div className="min-h-screen">
      <header className="mx-auto flex max-w-5xl items-center justify-between px-6 pt-10 pb-4">
        <div className="flex items-center gap-2.5">
          <span className="h-2 w-2 rounded-full bg-[var(--accent)] shadow-[0_0_10px_var(--accent)]" />
          <span className="font-mono text-xs tracking-[0.2em] text-[var(--text-faint)] uppercase">
            Deep Restore
          </span>
        </div>
        <a
          href="https://github.com/Bhavik-03/Image-Enhancement"
          className="font-mono text-xs text-[var(--text-faint)] transition-colors hover:text-[var(--text-muted)]"
        >
          Shallow-UWnet
        </a>
      </header>

      <main className="mx-auto max-w-5xl px-6 pb-24">
        <section className="animate-rise py-10 text-center sm:py-16">
          <h1 className="font-display mx-auto max-w-2xl text-4xl leading-[1.08] font-light text-balance sm:text-5xl">
            Give your dive footage back its <em className="text-[var(--accent)] not-italic">colour</em>.
          </h1>
          <p className="mx-auto mt-4 max-w-md text-[var(--text-muted)]">
            Water absorbs red light first. This restores it — fine-tuned on real underwater
            photos, frame by frame on video.
          </p>
        </section>

        {errorMsg && stage !== "processing" && (
          <p className="mb-4 rounded-xl border border-[var(--danger)]/30 bg-[var(--danger-soft)] px-4 py-3 text-sm text-[var(--danger)]">
            {errorMsg}
          </p>
        )}

        <AnimatePresence mode="wait">
          {stage === "idle" && (
            <motion.div key="upload" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <UploadZone onFile={handleFile} />
            </motion.div>
          )}

          {(stage === "ready" || stage === "processing") && file && kind && (
            <motion.div
              key="ready"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="space-y-5"
            >
              <div className="overflow-hidden rounded-2xl border border-[var(--border)] bg-[var(--bg-elevated)] p-3">
                <p className="mb-2 font-mono text-[11px] tracking-wide text-[var(--text-faint)] uppercase">
                  {file.name}
                </p>
                {kind === "image" ? (
                  <img src={sourceUrl!} alt="Selected" className="max-h-96 w-full rounded-xl object-contain" />
                ) : (
                  <video src={sourceUrl!} controls className="max-h-96 w-full rounded-xl bg-black" />
                )}
              </div>

              <div className="flex flex-wrap items-center justify-between gap-4">
                <ModelPicker models={models} value={model} onChange={setModel} defaultModel={defaultModel} />
                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    onClick={reset}
                    disabled={stage === "processing"}
                    className="text-sm text-[var(--text-faint)] transition-colors hover:text-[var(--text-muted)] disabled:opacity-40"
                  >
                    Choose another
                  </button>
                  <button
                    type="button"
                    onClick={runEnhance}
                    disabled={stage === "processing"}
                    className="flex items-center gap-2 rounded-full bg-[var(--accent)] px-5 py-2.5 text-sm font-medium text-[#1a0905] transition-colors hover:bg-[var(--accent-strong)] disabled:opacity-60"
                  >
                    {stage === "processing" ? (
                      <>
                        {kind === "video" ? (
                          <ProgressRing progress={videoStatus?.progress ?? null} size={18} />
                        ) : (
                          <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-[#1a0905]/30 border-t-[#1a0905]" />
                        )}
                        {kind === "video" && videoStatus?.progress != null
                          ? `Enhancing · ${Math.round(videoStatus.progress * 100)}%`
                          : "Enhancing…"}
                      </>
                    ) : (
                      "Enhance"
                    )}
                  </button>
                </div>
              </div>
            </motion.div>
          )}

          {stage === "done" && file && kind && resultUrl && sourceUrl && (
            <motion.div
              key="done"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-5"
            >
              {kind === "image" ? (
                <CompareSlider beforeSrc={sourceUrl} afterSrc={resultUrl} />
              ) : (
                <VideoCompare beforeSrc={sourceUrl} afterSrc={resultUrl} />
              )}
              <div className="flex items-center justify-between">
                <button
                  type="button"
                  onClick={reset}
                  className="text-sm text-[var(--text-faint)] transition-colors hover:text-[var(--text-muted)]"
                >
                  ← Enhance another
                </button>
                <a
                  href={resultUrl}
                  download={`enhanced-${file.name}`}
                  className="rounded-full border border-[var(--border-strong)] bg-[var(--surface)] px-5 py-2.5 text-sm font-medium transition-colors hover:bg-[var(--surface-hover)]"
                >
                  Download result
                </a>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}
