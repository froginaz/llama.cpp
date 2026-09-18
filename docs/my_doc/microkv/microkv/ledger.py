"""microkv.ledger - 아주 작은 KV 캐시 장부(ledger) 모형.

llama.cpp의 llama_kv_cells를 교육용으로 축소한 것으로, 셀 배정(assign),
범위 삭제(remove), 인과 마스크 파생(mask)만 제공한다.

모든 메서드는 결정론적이고 외부 의존성(I/O, 시간, 난수)이 없다.
"""

from __future__ import annotations

from typing import Optional


class CacheFullError(Exception):
    """빈 셀이 없어 배정에 실패했을 때 발생한다."""


class CellLedger:
    """고정 크기 셀 배열에 대한 (seq_id, pos) 장부.

    각 셀은 비어 있거나(None), 정확히 하나의 (seq_id, pos) 쌍을 가진다.
    셀의 데이터(K/V)는 이 모형의 관심사가 아니다 - 장부와 마스크만 다룬다.

    Example:
        >>> led = CellLedger(4)
        >>> led.assign(0, 0)
        0
        >>> led.assign(0, 1)
        1
        >>> led.mask(0, query_pos=0)
        [True, False, False, False]
    """

    def __init__(self, n_cells: int) -> None:
        """n_cells개의 빈 셀로 장부를 만든다.

        Args:
            n_cells: 셀 개수. 1 이상이어야 한다.

        Raises:
            ValueError: n_cells가 1 미만일 때.
        """
        if n_cells < 1:
            raise ValueError(f"n_cells must be >= 1, got {n_cells}")
        self.n_cells = n_cells
        self._pos: list[Optional[int]] = [None] * n_cells
        self._seq: list[Optional[int]] = [None] * n_cells

    @property
    def used(self) -> int:
        """현재 차 있는 셀의 개수."""
        return sum(1 for p in self._pos if p is not None)

    def assign(self, seq_id: int, pos: int) -> int:
        """번호가 가장 낮은 빈 셀에 (seq_id, pos)를 기록하고 그 셀 번호를 돌려준다.

        Args:
            seq_id: 시퀀스 번호 (0 이상).
            pos: 토큰 위치 (0 이상).

        Returns:
            배정된 셀 번호 (0 <= cell < n_cells).

        Raises:
            ValueError: seq_id 또는 pos가 음수일 때, 또는 같은 seq_id에
                같은 pos가 이미 존재할 때.
            CacheFullError: 빈 셀이 하나도 없을 때.
        """
        if seq_id < 0:
            raise ValueError(f"seq_id must be >= 0, got {seq_id}")
        if pos < 0:
            raise ValueError(f"pos must be >= 0, got {pos}")
        for s, p in zip(self._seq, self._pos):
            if s == seq_id and p == pos:
                raise ValueError(f"duplicate pos {pos} for seq {seq_id}")
        for i in range(self.n_cells):
            if self._pos[i] is None:
                self._pos[i] = pos
                self._seq[i] = seq_id
                return i
        raise CacheFullError(f"all {self.n_cells} cells are in use")

    def remove(self, seq_id: int, p0: int, p1: int = -1) -> int:
        """seq_id의 pos가 [p0, p1) 범위인 셀을 모두 비우고, 비운 개수를 돌려준다.

        p1이 -1이면 '끝까지'를 뜻한다 (p0 이상 전부).
        해당하는 셀이 없으면 0을 돌려준다 (에러가 아님).

        Args:
            seq_id: 시퀀스 번호.
            p0: 범위 시작 (포함).
            p1: 범위 끝 (미포함). -1이면 무한대로 취급.

        Returns:
            비워진 셀의 개수.

        Raises:
            ValueError: p0가 음수이거나, p1이 -1이 아니면서 p0보다 작을 때.
        """
        if p0 < 0:
            raise ValueError(f"p0 must be >= 0, got {p0}")
        if p1 != -1 and p1 < p0:
            raise ValueError(f"p1 ({p1}) must be -1 or >= p0 ({p0})")
        hi = float("inf") if p1 == -1 else p1
        removed = 0
        for i in range(self.n_cells):
            p = self._pos[i]
            if self._seq[i] == seq_id and p is not None and p0 <= p < hi:
                self._pos[i] = None
                self._seq[i] = None
                removed += 1
        return removed

    def mask(self, seq_id: int, query_pos: int) -> list[bool]:
        """query_pos 위치의 쿼리가 '볼 수 있는' 셀을 셀별 bool 리스트로 돌려준다.

        규칙 (인과 마스크): 셀이 같은 seq_id에 속하고, 그 셀의 pos가
        query_pos 이하일 때만 True. 빈 셀과 다른 seq의 셀은 항상 False.

        Args:
            seq_id: 쿼리가 속한 시퀀스 번호.
            query_pos: 쿼리 토큰의 위치 (0 이상).

        Returns:
            길이 n_cells의 bool 리스트.

        Raises:
            ValueError: query_pos가 음수일 때.
        """
        if query_pos < 0:
            raise ValueError(f"query_pos must be >= 0, got {query_pos}")
        return [
            s == seq_id and p is not None and p <= query_pos
            for s, p in zip(self._seq, self._pos)
        ]

    def pos_range(self, seq_id: int) -> Optional[tuple[int, int]]:
        """seq_id가 가진 셀들의 (최소 pos, 최대 pos)를 돌려준다.

        Args:
            seq_id: 시퀀스 번호.

        Returns:
            (min_pos, max_pos) 튜플. 해당 seq의 셀이 하나도 없으면 None.
        """
        ps = [p for s, p in zip(self._seq, self._pos) if s == seq_id and p is not None]
        if not ps:
            return None
        return (min(ps), max(ps))
