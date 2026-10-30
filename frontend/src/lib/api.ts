// Thin client for api.py. Dev proxies /api to the FastAPI server (see vite.config.ts);
// in production api.py serves this build itself, so relative paths work either way.

export interface ModelsResponse {
  models: string[];
  default: string;
}

export interface VideoJobStatus {
  status: "running" | "done" | "error";
  progress: number | null; // null when the source codec didn't report a frame count
  frames_done: number;
  error?: string;
}

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export function listModels(): Promise<ModelsResponse> {
  return fetch("/api/models").then(asJson<ModelsResponse>);
}

export async function enhancePhoto(file: File, model: string): Promise<Blob> {
  const form = new FormData();
  form.append("file", file);
  form.append("model", model);
  const res = await fetch("/api/enhance/photo", { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.blob();
}

export async function startVideoJob(
  file: File,
  model: string,
  smooth: number,
): Promise<string> {
  const form = new FormData();
  form.append("file", file);
  form.append("model", model);
  form.append("smooth", String(smooth));
  const res = await fetch("/api/enhance/video", { method: "POST", body: form });
  const body = await asJson<{ job_id: string }>(res);
  return body.job_id;
}

export function getVideoJobStatus(jobId: string): Promise<VideoJobStatus> {
  return fetch(`/api/enhance/video/${jobId}`).then(asJson<VideoJobStatus>);
}

export function videoResultUrl(jobId: string): string {
  return `/api/enhance/video/${jobId}/result`;
}

/** Polls a video job until it finishes or fails, reporting progress along the way. */
export async function pollVideoJob(
  jobId: string,
  onProgress: (status: VideoJobStatus) => void,
  intervalMs = 600,
): Promise<void> {
  for (;;) {
    const status = await getVideoJobStatus(jobId);
    onProgress(status);
    if (status.status === "done") return;
    if (status.status === "error") throw new Error(status.error ?? "video job failed");
    await new Promise((r) => setTimeout(r, intervalMs));
  }
}
