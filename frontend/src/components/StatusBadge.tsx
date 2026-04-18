type StatusTone = "success" | "warning" | "danger" | "neutral";

const CLASS_NAME: Record<StatusTone, string> = {
  success: "status-badge status-badge--success",
  warning: "status-badge status-badge--warning",
  danger: "status-badge status-badge--danger",
  neutral: "status-badge status-badge--neutral",
};

type StatusBadgeProps = {
  label: string;
  tone?: StatusTone;
};

export function StatusBadge({ label, tone = "neutral" }: StatusBadgeProps) {
  return <span className={CLASS_NAME[tone]}>{label}</span>;
}
