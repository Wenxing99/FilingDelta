import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";

import { apiBaseUrl } from "../../lib/api";
import type { CitationTarget, DemoDocument } from "../../lib/types";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).toString();

const pdfDocumentOptions = {
  cMapUrl: "/cmaps/",
  cMapPacked: true,
  standardFontDataUrl: "/standard_fonts/",
};

const BASE_PAGE_WIDTH = 860;
const MIN_MANUAL_SCALE = 0.7;
const MAX_MANUAL_SCALE = 2.2;
const SCALE_STEP = 0.12;

type PdfDocumentViewerProps = {
  document: DemoDocument;
  citationTarget: CitationTarget | null;
};

export function PdfDocumentViewer({ document, citationTarget }: PdfDocumentViewerProps) {
  const [pageCount, setPageCount] = useState(0);
  const [currentPage, setCurrentPage] = useState(citationTarget?.page ?? 1);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [viewportWidth, setViewportWidth] = useState(0);
  const [zoomMode, setZoomMode] = useState<"fit" | "manual">("fit");
  const [manualScale, setManualScale] = useState(1);

  const viewportRef = useRef<HTMLDivElement | null>(null);
  const pageRefs = useRef(new Map<number, HTMLDivElement>());

  const syncCurrentPageFromScroll = useCallback(() => {
    const viewportNode = viewportRef.current;
    if (!viewportNode) {
      return;
    }

    const pageEntries = Array.from(pageRefs.current.entries()).sort((left, right) => left[0] - right[0]);
    if (pageEntries.length === 0) {
      return;
    }

    const threshold = viewportNode.scrollTop + 24;
    let visiblePage = pageEntries[0][0];

    for (const [pageNumber, pageNode] of pageEntries) {
      if (pageNode.offsetTop <= threshold) {
        visiblePage = pageNumber;
      } else {
        break;
      }
    }

    setCurrentPage((previous) => (previous === visiblePage ? previous : visiblePage));
  }, []);

  useEffect(() => {
    setCurrentPage(citationTarget?.page ?? 1);
  }, [citationTarget?.page, document.document_id]);

  useEffect(() => {
    setPageCount(0);
    setLoadError(null);
    setCurrentPage(citationTarget?.page ?? 1);
    pageRefs.current.clear();
    viewportRef.current?.scrollTo({ top: 0, behavior: "auto" });
  }, [document.document_id]);

  useEffect(() => {
    const viewportNode = viewportRef.current;
    if (!viewportNode) {
      return undefined;
    }

    const updateWidth = () => {
      setViewportWidth(viewportNode.clientWidth);
    };

    updateWidth();

    const observer = new ResizeObserver(() => {
      updateWidth();
    });

    observer.observe(viewportNode);
    return () => observer.disconnect();
  }, [document.document_id]);

  useEffect(() => {
    const viewportNode = viewportRef.current;
    if (!viewportNode) {
      return undefined;
    }

    syncCurrentPageFromScroll();
    viewportNode.addEventListener("scroll", syncCurrentPageFromScroll, { passive: true });

    return () => viewportNode.removeEventListener("scroll", syncCurrentPageFromScroll);
  }, [document.document_id, pageCount, syncCurrentPageFromScroll]);

  useEffect(() => {
    const targetPage = citationTarget?.page;
    if (!targetPage) {
      return;
    }

    scrollToPage(targetPage, "smooth");
  }, [citationTarget?.page, pageCount, document.document_id]);

  const sourceUrl = `${apiBaseUrl()}${document.source_url}`;

  const renderedWidth = useMemo(() => {
    if (zoomMode === "fit") {
      return Math.max(320, Math.floor(viewportWidth - 72));
    }
    return Math.round(BASE_PAGE_WIDTH * manualScale);
  }, [manualScale, viewportWidth, zoomMode]);

  const zoomPercent = useMemo(() => {
    const safeWidth = renderedWidth > 0 ? renderedWidth : BASE_PAGE_WIDTH;
    return Math.round((safeWidth / BASE_PAGE_WIDTH) * 100);
  }, [renderedWidth]);

  useEffect(() => {
    if (pageCount === 0) {
      return undefined;
    }

    const frame = window.requestAnimationFrame(() => {
      syncCurrentPageFromScroll();
    });

    return () => window.cancelAnimationFrame(frame);
  }, [document.document_id, pageCount, renderedWidth, syncCurrentPageFromScroll]);

  function adjustZoom(delta: number) {
    setZoomMode("manual");
    setManualScale((value) => clamp(value + delta, MIN_MANUAL_SCALE, MAX_MANUAL_SCALE));
  }

  function switchToFitWidth() {
    setZoomMode("fit");
  }

  function scrollToPage(targetPage: number, behavior: ScrollBehavior = "smooth") {
    const pageNode = pageRefs.current.get(targetPage);
    if (!pageNode) {
      return;
    }

    pageNode.scrollIntoView({ block: "start", behavior });
    setCurrentPage(targetPage);
  }

  return (
    <div className="pdf-viewer">
      <div className="pdf-viewer__toolbar">
        <div className="pdf-viewer__status">
          <span>{pageCount > 0 ? `连续阅读 · 第 ${currentPage} / ${pageCount} 页` : "正在加载 PDF"}</span>
        </div>

        <div className="pdf-viewer__controls">
          <div className="pdf-viewer__control-group">
            <button
              type="button"
              className="viewer-action"
              disabled={currentPage <= 1}
              onClick={() => scrollToPage(Math.max(1, currentPage - 1))}
            >
              上一页
            </button>
            <select
              className="viewer-page-select"
              value={pageCount > 0 ? currentPage : ""}
              disabled={pageCount === 0}
              onChange={(event) => scrollToPage(Number(event.target.value))}
            >
              {Array.from({ length: pageCount }, (_, index) => {
                const pageNumber = index + 1;
                return (
                  <option key={pageNumber} value={pageNumber}>
                    第 {pageNumber} 页
                  </option>
                );
              })}
            </select>
            <button
              type="button"
              className="viewer-action"
              disabled={pageCount === 0 || currentPage >= pageCount}
              onClick={() => scrollToPage(Math.min(pageCount, currentPage + 1))}
            >
              下一页
            </button>
          </div>

          <div className="pdf-viewer__control-group">
            <button
              type="button"
              className="viewer-action"
              onClick={() => adjustZoom(-SCALE_STEP)}
            >
              -
            </button>
            <button
              type="button"
              className={`viewer-action ${zoomMode === "fit" ? "viewer-action--active" : ""}`}
              onClick={switchToFitWidth}
            >
              适应宽度
            </button>
            <span className="pdf-viewer__zoom-label">{zoomPercent}%</span>
            <button
              type="button"
              className="viewer-action"
              onClick={() => adjustZoom(SCALE_STEP)}
            >
              +
            </button>
          </div>
        </div>
      </div>

      <div ref={viewportRef} className="pdf-viewer__canvas-shell">
        <Document
          key={sourceUrl}
          file={sourceUrl}
          options={pdfDocumentOptions}
          loading={<div className="viewer-empty">正在加载 PDF...</div>}
          onLoadSuccess={({ numPages }) => {
            setPageCount(numPages);
            setLoadError(null);
            setCurrentPage((value) => Math.min(Math.max(1, value), numPages));
          }}
          onLoadError={(error) => {
            setLoadError(error.message);
          }}
        >
          {loadError ? (
            <div className="viewer-empty">
              <p>PDF 加载失败。</p>
              <p>{loadError}</p>
            </div>
          ) : (
            <div className="pdf-viewer__page-stack">
              {Array.from({ length: pageCount }, (_, index) => {
                const pageNumber = index + 1;
                const isActivePage = citationTarget?.page === pageNumber;

                return (
                  <div
                    key={`${document.document_id}:page:${pageNumber}`}
                    ref={(node) => {
                      if (node) {
                        pageRefs.current.set(pageNumber, node);
                      } else {
                        pageRefs.current.delete(pageNumber);
                      }
                    }}
                    className={`pdf-viewer__page-card ${isActivePage ? "pdf-viewer__page-card--active" : ""}`}
                    data-page-number={pageNumber}
                  >
                    <div className="pdf-viewer__page-label">第 {pageNumber} 页</div>
                    <Page
                      pageNumber={pageNumber}
                      renderAnnotationLayer={false}
                      renderTextLayer={false}
                      width={renderedWidth}
                      loading={<div className="viewer-empty">正在渲染第 {pageNumber} 页...</div>}
                    />
                  </div>
                );
              })}
            </div>
          )}
        </Document>
      </div>
    </div>
  );
}

function clamp(value: number, lower: number, upper: number) {
  return Math.min(Math.max(value, lower), upper);
}
