package exp_unit

import chisel3._
import chisel3.util._

import chisel3.experimental.FixedPoint

import math.pow

class ExpUnitFixPoint(width: Int, point: Int, lut_bits: Int, append_bits: Int)
    extends Module {
  val v_width = width + append_bits
  val v_point = point + append_bits
  val fpType = FixedPoint(width.W, point.BP)
  val vType = FixedPoint(v_width.W, v_point.BP)
  val io = IO(new Bundle {
    val in_value = Input(fpType)
    val out_exp = Output(fpType)
  })

  val x = Wire(UInt(width.W))
  val y = Wire(UInt(v_width.W))
  val z1 = Wire(vType)
  val z2 = Wire(vType)

  val s = Reg(fpType)

  val u = Wire(UInt((width - point).W))
  val v = Wire(vType)

  val testers =
    Range.Double(0.0, 1.0, pow(2.0, -point)).map((a) => pow(2.0, a) - a)
  val d_value =
    (testers.reduce((a, b) => if (a > b) a else b) +
      testers.reduce((a, b) => if (a < b) a else b)) / 2.0

  val d_fixed = FixedPoint.fromDouble(d_value, v_width.W, v_point.BP)
  val d_wire = Wire(vType)
  if (lut_bits == 0)
    d_wire := d_fixed
  else {
    val lut_in = Range(0, 1 << lut_bits)
    val lut_out =
      lut_in
        .map((x) => x / pow(2.0, lut_bits))
        .map((x) => {
          val r = Range
            .Double(x, x + pow(2.0, -lut_bits), pow(2.0, -lut_bits))
            .map((y) => pow(2.0, y) - y)
          (r.reduce((a, b) => if (a > b) a else b) +
            r.reduce((a, b) => if (a < b) a else b)) / 2.0
        })
        .map((x) =>
          FixedPoint
            .fromDouble(x, v_width.W, v_point.BP)
        )
    // val lut_mem = Mem(lut_in.length, vType)
    // for (i <- 0 until lut_out.length)
    //   lut_mem(i.U) := lut_out(i)

    val v_bits = Wire(UInt(lut_bits.W))
    v_bits := v.asUInt()(
      v_point - 1,
      v_point - lut_bits
    )

    var w = when(v_bits === lut_in(0).U) {
      d_wire := lut_out(0)
    }
    for (i <- 1 until lut_in.size)
      w = w.elsewhen(v_bits === lut_in(i).U) {
        d_wire := lut_out(i)
      }
    w.otherwise {
      d_wire := DontCare
    }
    // d_wire := lut_mem(v_bits)
  }
  // println(d_fixed)

  x := io.in_value.asUInt()
  y := (x << append_bits) + (x << (append_bits - 1)) - (x << (append_bits - 4));

  u := y(v_width - 1, v_point)
  v := Cat(0.U((v_width - v_point).W), y(v_point - 1, 0))
    .asFixedPoint(v_point.BP)

  z1 := v + d_wire
  z2 := z1 << u;

  // printf(
  //   "x:%b y:%b u:%b v:%b d:%b z1:%b z2:%b\n",
  //   x,
  //   y,
  //   u,
  //   v.asUInt(),
  //   d_wire.asUInt(),
  //   z1.asUInt(),
  //   z2.asUInt()
  // )

  io.out_exp := z2
}
