package pe_row

import chisel3._
import chisel3.util._

import chisel3.experimental.FixedPoint

import exp_unit.ExpUnitFixPoint

class DensePE(outBits: Int) extends Module {
  val io = IO(new Bundle {
    val l_in = Input(UInt(4.W))
    val t_in = Input(UInt(4.W))
    val r_out = Output(UInt(4.W))
    val b_out = Output(UInt(4.W))
    val s_in = Input(UInt(outBits.W))
    val s_out = Output(UInt(outBits.W))

    val c_state = Input(UInt(2.W))
  })

  val idle :: clear :: calc :: move :: Nil = Enum(4)

  val v_reg = Reg(UInt(4.W))
  val h_reg = Reg(UInt(4.W))

  val s_reg = Reg(UInt(outBits.W))

  io.b_out := v_reg
  io.r_out := h_reg
  io.s_out := s_reg

  switch(io.c_state) {
    is(idle) {
      v_reg := v_reg
      h_reg := h_reg
      s_reg := s_reg
    }
    is(clear) {
      v_reg := 0.U
      h_reg := 0.U
      s_reg := 0.U
    }
    is(calc) {
      v_reg := io.t_in
      h_reg := io.l_in
      s_reg := s_reg + v_reg * h_reg
    }
    is(move) {
      s_reg := io.s_in
      v_reg := v_reg
      h_reg := h_reg
    }
  }
}

class Dense_PE_Array(
    bits: Int,
    point: Int,
    append: Int,
    internalBits: Int,
    width: Int,
    height: Int
) extends Module {
  val fpType = FixedPoint(bits.W, point.BP)
  val io = IO(new Bundle {
    val l_in = Input(Vec(height, UInt(4.W)))
    val t_in = Input(Vec(width, UInt(4.W)))
    val s_out = Output(Vec(height, fpType))
    val exp_sum = Output(Vec(height, fpType))
    val c_pestate = Input(UInt(2.W))
    val c_astate = Input(UInt(2.W))
  })

  val a_idle :: a_clear :: a_calc :: Nil = Enum(3)

  val pes =
    (for (i <- 0 until height)
      yield for (j <- 0 until width) yield Module(new DensePE(internalBits)))

  val pe_idle = pes(0)(0).idle
  val pe_clear = pes(0)(0).clear
  val pe_calc = pes(0)(0).calc
  val pe_move = pes(0)(0).move

  for (i <- 0 until height) {
    pes(i)(0).io.l_in := io.l_in(i)
    pes(i)(0).io.s_in := 0.U
  }
  for (i <- 0 until width)
    pes(0)(i).io.t_in := io.t_in(i)

  for (i <- 0 until height)
    for (j <- 1 until width) {
      pes(i)(j).io.l_in := pes(i)(j - 1).io.r_out
      pes(i)(j).io.s_in := pes(i)(j - 1).io.s_out
    }

  for (i <- 1 until height)
    for (j <- 0 until width)
      pes(i)(j).io.t_in := pes(i - 1)(j).io.b_out

  for (i <- 0 until height)
    for (j <- 0 until width)
      pes(i)(j).io.c_state := io.c_pestate

  val exps =
    for (i <- 0 until height)
      yield Module(
        new ExpUnitFixPoint(bits, point, 6, 4)
      )
  val sum_regs = Reg(Vec(height, fpType))
  val exp_regs = Reg(Vec(height, fpType))

  io.s_out := exp_regs
  io.exp_sum := sum_regs

  for (i <- 0 until height) {
    if (append > 0)
      exps(i).io.in_value := Cat(
        0.U((bits - internalBits - append).W),
        pes(i)(width - 1).io.s_out,
        0.U(append.W)
      ).asFixedPoint(point.BP)
    else
      exps(i).io.in_value := Cat(
        0.U((bits - internalBits).W),
        pes(i)(width - 1).io.s_out
      ).asFixedPoint(point.BP)
    exp_regs(i) := exps(i).io.out_exp
  }
  switch(io.c_astate) {
    is(a_idle) {
      for (i <- 0 until height) sum_regs(i) := sum_regs(i)
    }
    is(a_clear) {
      for (i <- 0 until height) sum_regs(i) := 0.F(bits.W, point.BP)
    }
    is(a_calc) {
      for (i <- 0 until height) sum_regs(i) := sum_regs(i) + exp_regs(i)
    }
  }
}
