package pe_row

import chisel3._
import chisel3.util._

import chisel3.experimental.FixedPoint

import exp_unit.ExpUnitFixPoint

object SeqSwitch {
  def apply[T <: Data](sel: UInt, out: T, cases: IndexedSeq[Tuple2[UInt, T]]) {
    var w = when(sel === cases.head._1) {
      out := cases.head._2
    }
    for ((u, v) <- cases.tail) {
      w = w.elsewhen(sel === u) {
        out := v
      }
    }
    w.otherwise {
      out := DontCare
    }
  }
}

// Controls:
// c1: input from row / score_exp
// c2: input from col / 0
// c3: stage1/stage2
// c4: bubble
// c5: exp unit

class PE(
    bits: Int,
    point: Int,
    width: Int,
    buf_size: Int,
    c1_bits: Int,
    c2_bits: Int,
    c4_bits: Int,
    id: (Int, Int)
) extends Module {
  val fpType = FixedPoint(bits.W, point.BP)
  val io = IO(new Bundle {
    val row_in = Input(Vec(width, fpType))
    val col_in = Input(Vec(width, fpType))
    val o_in = Input(fpType)
    val o_out = Output(fpType)
    val score_in = Input(fpType)
    val score_out = Output(fpType)

    val c1 = Input(UInt(c1_bits.W))
    val c2 = Input(UInt(c2_bits.W))
    val c3 = Input(UInt(2.W))
    val c4 = Input(UInt(c4_bits.W))
    val c5 = Input(UInt(2.W))
  })

  val acc_clear :: acc_idle :: acc_self :: acc_left :: Nil = Enum(4); // c3
  val exp_idle :: exp_calc :: exp_move :: Nil = Enum(3); // c5

  val exp_unit = Module(new ExpUnitFixPoint(bits, point, 6, 4))

  val a = Wire(fpType)
  val b = Wire(fpType)
  val c = Wire(fpType)
  val acc = Reg(fpType)
  val score_exp = Reg(fpType)

  // Buffer
  val buf = Reg(Vec(buf_size, fpType))
  val buf_vec = Wire(Vec(buf_size + 1, fpType))
  buf_vec(0) := acc
  for (i <- 0 until buf_size) buf_vec(i + 1) := buf(i)

  // MAC
  val a_vec = Wire(Vec(width + 2, fpType))
  for (i <- 0 until width) a_vec(i) := io.row_in(i)
  a_vec(width) := score_exp
  a_vec(width + 1) := FixedPoint(0, bits.W, point.BP)
  SeqSwitch(
    io.c1,
    a,
    (Range(0, width + 2).map(_.U)).zip(a_vec)
  )
  val b_vec = Wire(Vec(width + 1, fpType))
  for (i <- 0 until width) b_vec(i) := io.col_in(i)
  b_vec(width) := FixedPoint(0, bits.W, point.BP)
  SeqSwitch(
    io.c2,
    b,
    Range(0, width + 1).map(_.U).zip(b_vec)
  )
  c := a * b

  // Score
  io.score_out := score_exp

  // Control
  io.o_out := DontCare
  switch(io.c3) {
    is(acc_clear) {
      acc := FixedPoint(0, bits.W, point.BP)
      for (i <- 0 until buf.size)
        buf(i) := FixedPoint(0, bits.W, point.BP)
    }
    is(acc_idle) {
      acc := acc
      for (i <- 0 until buf.size)
        buf(i) := buf(i)
    }
    is(acc_self) {
      acc := acc + c
      for (i <- 0 until buf.size)
        buf(i) := buf(i)
    }
    is(acc_left) {
      acc := io.o_in + c

      buf(0) := acc
      for (i <- 0 until buf_size - 1)
        buf(i + 1) := buf(i)
      SeqSwitch(
        io.c4,
        io.o_out,
        Range(0, buf_size + 1).map(_.U).zip(buf_vec)
      )
    }
  }
  exp_unit.io.in_value := FixedPoint(0, bits.W, point.BP)
  switch(io.c5) {
    is(exp_idle) {
      score_exp := score_exp
    }
    is(exp_calc) {
      exp_unit.io.in_value := acc
      score_exp := exp_unit.io.out_exp
    }
    is(exp_move) {
      score_exp := io.score_in
    }
  }
}

class PE_Row(
    pe_n: Int,
    bits: Int,
    point: Int,
    width: Int,
    c1_bits: Int,
    c2_bits: Int,
    c4_bits: Int,
    rowId: Int
) extends Module {
  val fpType = FixedPoint(bits.W, point.BP)
  val io = IO(new Bundle {
    val left_in = Input(fpType)
    val top_in = Input(Vec(width, fpType))
    val bot_out = Output(Vec(width, fpType))
    val o_out = Output(fpType)
    val score_sum = Output(fpType)

    val clr = Input(Bool())
    val c1 = Input(Vec(pe_n, UInt(c1_bits.W)))
    val c2 = Input(Vec(pe_n, UInt(c2_bits.W)))
    val c3 = Input(UInt(2.W))
    val c4 = Input(Vec(pe_n, UInt(c4_bits.W)))
    val c5 = Input(UInt(2.W))
  })

  val pes =
    for (i <- 0 until pe_n)
      yield Module(
        new PE(
          bits,
          point,
          width,
          width - pe_n,
          c1_bits,
          c2_bits,
          c4_bits,
          (rowId, i)
        )
      )

  val (acc_clear, acc_idle, acc_self, acc_left) = (
    pes(0).acc_clear,
    pes(0).acc_idle,
    pes(0).acc_self,
    pes(0).acc_left
  )

  val (exp_idle, exp_calc, exp_move) = (
    pes(0).exp_idle,
    pes(0).exp_calc,
    pes(0).exp_move
  );

  val v_regs = Reg(Vec(width, fpType))
  val h_regs = Reg(Vec(width, fpType))

  val ssum = Reg(fpType)
  io.score_sum := ssum

  for (i <- 0 until pe_n) {
    pes(i).io.row_in := v_regs
    pes(i).io.col_in := h_regs
    pes(i).io.c1 := io.c1(i)
    pes(i).io.c2 := io.c2(i)
    pes(i).io.c3 := io.c3
    pes(i).io.c4 := io.c4(i)
    pes(i).io.c5 := io.c5
  }
  for (i <- 0 until pe_n - 1) {
    pes(i + 1).io.score_in := pes(i).io.score_out
    pes(i + 1).io.o_in := pes(i).io.o_out
  }
  pes(0).io.score_in := FixedPoint(0, bits.W, point.BP)
  pes(0).io.o_in := FixedPoint(0, bits.W, point.BP)

  for (i <- 0 until width)
    io.bot_out(i) := h_regs(i)

  io.o_out := pes(pe_n - 1).io.o_out

  when(io.clr) {
    for (i <- 0 until width) {
      v_regs(i) := FixedPoint(0, bits.W, point.BP)
      h_regs(i) := FixedPoint(0, bits.W, point.BP)
    }
    ssum := FixedPoint(0, bits.W, point.BP)
  }.otherwise {
    for (i <- 0 until width - 1)
      v_regs(i + 1) := v_regs(i)
    v_regs(0) := io.left_in

    h_regs := io.top_in
    io.bot_out := h_regs

    when(io.c5 === exp_move) {
      ssum := ssum + pes(pe_n - 1).io.score_out
    }.otherwise {
      ssum := ssum
    }
  }
}

class Sparse_PE_Array(
    pe_n: Int,
    bits: Int,
    point: Int,
    width: Int,
    height: Int,
    c1_bits: Int,
    c2_bits: Int,
    c4_bits: Int
) extends Module {
  val fpType = FixedPoint(bits.W, point.BP)
  val io = IO(new Bundle {
    val left_in = Input(Vec(height, fpType))
    val top_in = Input(Vec(width, fpType))
    val out = Output(Vec(height, fpType))
    val score_sum = Output(Vec(height, fpType))
    val clr = Input(Bool())
    val c1 = Input(Vec(height, Vec(pe_n, UInt(c1_bits.W))))
    val c2 = Input(Vec(height, Vec(pe_n, UInt(c2_bits.W))))
    val c3 = Input(UInt(2.W))
    val c4 = Input(Vec(height, Vec(pe_n, UInt(c4_bits.W))))
    val c5 = Input(UInt(2.W))
  })
  val rows =
    for (i <- 0 until height)
      yield Module(
        new PE_Row(
          pe_n,
          bits,
          point,
          width,
          c1_bits,
          c2_bits,
          c4_bits,
          i
        )
      )
  val (acc_clear, acc_idle, acc_self, acc_left) = (
    rows(0).acc_clear,
    rows(0).acc_idle,
    rows(0).acc_self,
    rows(0).acc_left
  )

  val (exp_idle, exp_calc, exp_move) = (
    rows(0).exp_idle,
    rows(0).exp_calc,
    rows(0).exp_move
  );

  for (i <- 0 until height) {
    rows(i).io.clr := io.clr
    rows(i).io.c1 := io.c1(i)
    rows(i).io.c2 := io.c2(i)
    rows(i).io.c3 := io.c3
    rows(i).io.c4 := io.c4(i)
    rows(i).io.c5 := io.c5
    rows(i).io.left_in := io.left_in(i)
    io.out(i) := rows(i).io.o_out
    io.score_sum(i) := rows(i).io.score_sum
  }
  rows(0).io.top_in := io.top_in
  for (i <- 0 until height - 1)
    rows(i + 1).io.top_in := rows(i).io.bot_out
}
