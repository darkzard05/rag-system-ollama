"""
UI 스트리밍 통합 - Streamlit과 스트리밍 처리기 연동
"""

import logging
import time

import streamlit as st
from langchain_core.documents import Document

from api.streaming_handler import (
    AdaptiveStreamingController,
    StreamChunk,
    StreamingResponseBuilder,
    StreamingResponseHandler,
    get_adaptive_controller,
    get_streaming_handler,
)

logger = logging.getLogger(__name__)


class StreamlitStreamingUI:
    """
    Streamlit 기반 스트리밍 UI

    특징:
    - 실시간 응답 표시
    - 토큰 카운팅 및 속도 표시
    - 성능 메트릭 출력
    """

    def __init__(self):
        self.builder = StreamingResponseBuilder()
        self.streaming_handler: StreamingResponseHandler | None = None
        self.adaptive_controller: AdaptiveStreamingController | None = None

    async def stream_response_to_ui(
        self,
        response_generator,
        chat_container,
        show_metrics: bool = True,
        show_tokens_per_second: bool = True,
    ) -> str:
        """
        응답을 스트리밍으로 UI에 표시

        Args:
            response_generator: 토큰을 생성하는 비동기 이터레이터
            chat_container: Streamlit 채팅 컨테이너
            show_metrics: 성능 메트릭 표시 여부
            show_tokens_per_second: 토큰/초 표시 여부

        Returns:
            최종 응답 텍스트
        """
        self.streaming_handler = get_streaming_handler()
        self.adaptive_controller = get_adaptive_controller()
        self.builder = StreamingResponseBuilder()

        with chat_container, st.chat_message("assistant", avatar="🤖"):
            # 응답 표시 컨테이너
            response_container = st.empty()
            metrics_container = st.empty()

            # 응답 텍스트
            response_text = ""

            async def on_chunk(chunk: StreamChunk) -> None:
                """청크 수신 시 콜백"""
                nonlocal response_text

                response_text += chunk.content
                self.builder.add_chunk(chunk)

                # 메트릭 기록 (실제 지연 시간 계산)
                latency_ms = (time.time() - chunk.timestamp) * 1000
                if self.adaptive_controller:
                    self.adaptive_controller.record_latency(latency_ms)

                # UI 업데이트
                self._update_response_display(
                    response_container,
                    metrics_container,
                    response_text,
                    chunk.chunk_index,
                    show_metrics,
                    show_tokens_per_second,
                )

            async def on_complete() -> None:
                """스트리밍 완료 시 콜백"""
                if not self.streaming_handler:
                    return

                # 최종 메트릭 표시
                metrics = self.streaming_handler.metrics
                logger.info(
                    f"[Streaming Complete] "
                    f"총 토큰: {metrics.total_tokens}, "
                    f"처리 시간: {metrics.total_time:.2f}초, "
                    f"속도: {metrics.tokens_per_second:.1f} tok/s, "
                    f"첫 토큰 지연: {metrics.first_token_latency * 1000:.2f}ms"
                )

                # 최종 메트릭 표시
                if show_metrics:
                    self._display_final_metrics(metrics_container, metrics)

            async def on_error(error: Exception) -> None:
                """에러 발생 시 콜백"""
                error_msg = f"⚠️ 스트리밍 오류: {str(error)}"
                logger.error(error_msg)
                response_container.error(error_msg)

            # 스트리밍 실행
            await self.streaming_handler.stream_response(
                response_generator,
                on_chunk,
                on_complete,
                on_error,
                adaptive_controller=self.adaptive_controller,
            )

            return response_text

    def _update_response_display(
        self,
        response_container,
        metrics_container,
        response_text: str,
        chunk_index: int,
        show_metrics: bool,
        show_tokens_per_second: bool,
    ) -> None:
        """응답 표시 업데이트"""
        try:
            # 응답 텍스트 표시 (커서 애니메이션)
            response_container.markdown(response_text + " ▌", unsafe_allow_html=True)

            # 메트릭 표시
            if show_metrics and chunk_index % 10 == 0 and self.streaming_handler:
                metrics = self.streaming_handler.metrics

                if metrics.tokens_per_second > 0 and show_tokens_per_second:
                    metric_text = (
                        f"⏱️ {metrics.tokens_per_second:.1f} tok/s | "
                        f"📊 {metrics.total_tokens} 토큰 | "
                        f"⏰ {metrics.total_time:.1f}초"
                    )
                    metrics_container.caption(metric_text)

        except Exception as e:
            logger.error(f"[UI Update] 업데이트 오류: {e}")

    def _display_final_metrics(self, metrics_container, metrics) -> None:
        """최종 메트릭 표시"""
        try:
            metric_cols = metrics_container.columns(4)

            metric_cols[0].metric("📊 총 토큰", f"{metrics.total_tokens}개")

            metric_cols[1].metric("⏱️ 속도", f"{metrics.tokens_per_second:.1f} tok/s")

            metric_cols[2].metric("⏰ 처리 시간", f"{metrics.total_time:.2f}초")

            metric_cols[3].metric("📈 청크 수", f"{metrics.chunk_count}개")

            # 추가 메트릭
            with metrics_container.expander("📈 상세 메트릭"):
                detail_cols = st.columns(3)

                detail_cols[0].metric(
                    "첫 토큰 지연", f"{metrics.first_token_latency * 1000:.2f}ms"
                )

                detail_cols[1].metric(
                    "평균 청크 크기", f"{metrics.avg_chunk_size:.1f} 토큰"
                )

                detail_cols[2].metric(
                    "최소/최대 지연",
                    f"{metrics.min_latency * 1000:.1f}ms / {metrics.max_latency * 1000:.1f}ms",
                )

        except Exception as e:
            logger.error(f"[Metrics Display] 메트릭 표시 오류: {e}")


class DocumentCitationUI:
    """
    문서 인용 UI - 응답에서 문서 인용 강조
    """

    @staticmethod
    def format_response_with_citations(
        response_text: str, documents: list[Document]
    ) -> str:
        """
        응답 텍스트에 문서 인용 포맷팅

        Args:
            response_text: 원본 응답 텍스트
            documents: 참조 문서 리스트

        Returns:
            포맷팅된 HTML 텍스트
        """
        if not documents:
            return response_text

        # 문서 메타데이터 추출
        citations = {}
        for i, doc in enumerate(documents, 1):
            page = doc.metadata.get("page", "?")
            source = doc.metadata.get("source", "Unknown")
            citations[f"[p.{page}]"] = {"page": page, "source": source, "index": i}

        # 응답에 링크 추가
        html = response_text
        for citation_key, citation_info in citations.items():
            if citation_key in html:
                # 툴팁 추가
                tooltip = f"📄 {citation_info['source']} (p.{citation_info['page']})"
                link_html = (
                    f"<span title='{tooltip}' style='color: #0066cc; "
                    f"cursor: help; text-decoration: underline;'>"
                    f"{citation_key}</span>"
                )
                html = html.replace(citation_key, link_html)

        return html

    @staticmethod
    def display_document_panel(
        documents: list[Document], title: str = "📚 참고 문서"
    ) -> None:
        """
        문서 패널 표시

        Args:
            documents: 참조 문서 리스트
            title: 패널 제목
        """
        if not documents:
            st.info("참고 문서가 없습니다.")
            return

        with st.expander(f"{title} ({len(documents)}개)"):
            for i, doc in enumerate(documents, 1):
                col1, col2 = st.columns([0.2, 0.8])

                with col1:
                    st.caption(f"📄 {i}")

                with col2:
                    page = doc.metadata.get("page", "?")
                    source = doc.metadata.get("source", "Unknown")
                    st.caption(f"{source} (p.{page})")

                # 문서 내용 미리보기
                content = doc.page_content
                preview_length = 150
                preview = (
                    content[:preview_length] + "..."
                    if len(content) > preview_length
                    else content
                )
                st.text(preview)
                st.divider()


class StreamingMetricsDisplay:
    """
    스트리밍 성능 메트릭 표시
    """

    @staticmethod
    def display_streaming_metrics(metrics) -> None:
        """스트리밍 메트릭 표시"""
        st.subheader("📊 스트리밍 성능")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("총 토큰", f"{metrics.total_tokens}개")

        with col2:
            st.metric("처리 시간", f"{metrics.total_time:.2f}초")

        with col3:
            st.metric("속도", f"{metrics.tokens_per_second:.1f} tok/s")

        with col4:
            st.metric("첫 토큰 지연", f"{metrics.first_token_latency * 1000:.0f}ms")

    @staticmethod
    def display_adaptive_metrics(
        adaptive_controller: AdaptiveStreamingController,
    ) -> None:
        """적응형 제어 메트릭 표시"""
        metrics = adaptive_controller.get_metrics()

        if not metrics:
            return

        st.subheader("⚙️ 적응형 제어")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("평균 지연", f"{metrics['avg_latency_ms']:.1f}ms")

        with col2:
            st.metric("버퍼 크기", f"{metrics['current_buffer_size']}개")

        with col3:
            st.metric("샘플 수", f"{metrics['sample_count']:.0f}")


# 전역 인스턴스
_streamlit_ui: StreamlitStreamingUI | None = None


def get_streamlit_streaming_ui() -> StreamlitStreamingUI:
    """Streamlit 스트리밍 UI 인스턴스 반환"""
    global _streamlit_ui
    if _streamlit_ui is None:
        _streamlit_ui = StreamlitStreamingUI()
    return _streamlit_ui
