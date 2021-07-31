package pe_row

import chisel3._
import chisel3.util._

object RoundUp {
  def apply(x: Int) = {
    var y = 1
    var i = 0
    while (y < x) {
      y = y << 1
      i += 1
    }
    (y, i)
  }
}

class Counter(bits: Int) extends Module {
  val io = IO(new Bundle {
    val en = Input(Bool())
    val clr = Input(Bool())
    val out = Output(UInt(bits.W))
  })
  val r = RegInit(UInt(bits.W), 0.U)

  if (bits == 1)
    r := (~io.clr) & (r ^ io.en)
  else {
    val f = Wire(UInt(bits.W))
    val p =
      (for (i <- 1 until bits) yield (r(i - 1, 0).andR() & io.en).asUInt())
    f := Cat(
      p.reverse.reduce(Cat(_, _)),
      io.en.asUInt()
    )
    r := (Fill(bits, ~io.clr)) & (r ^ f)
  }

  io.out := r
}

class Pack(
    height: Int,
    pe_n: Int,
    mask_bits: Int,
    counter_bits: Int,
    row_id_bits: Int,
    no_pe: Int
) extends Module {
  val io = IO(new Bundle {
    val binaries = Input(Vec(height, Bool()))
    val mask = Output(Vec(height, UInt(mask_bits.W)))
    val emit = Output(Bool())
    val maskWrite = Output(Bool())
    val c_in = Input(UInt(2.W))
  })

  val in_idle :: in_clear :: in_read :: Nil = Enum(3);

  val mask_reg = Reg(Vec(height, UInt(mask_bits.W)))
  val write_reg = Reg(Bool())
  io.mask := mask_reg
  io.maskWrite := write_reg

  val b_counter =
    for (i <- 0 until height) yield Module(new Counter(counter_bits))
  val r_counter = Module(new Counter(row_id_bits))

  r_counter.io.en := 0.B
  r_counter.io.clr := 0.B
  for (i <- 0 until height) {
    b_counter(i).io.en := 0.B
    b_counter(i).io.clr := 0.B
  }
  io.emit := 0.B
  mask_reg := DontCare

  switch(io.c_in) {
    is(in_idle) {
      write_reg := write_reg
    }
    is(in_clear) {
      for (i <- 0 until height) {
        b_counter(i).io.en := 0.B
        b_counter(i).io.clr := 1.B
      }
      r_counter.io.en := 0.B
      r_counter.io.clr := 1.B
      for (i <- 0 until height)
        mask_reg(i) := 0.U
      io.emit := 0.B
      write_reg := 0.B
    }
    is(in_read) {
      val put =
        (for (b <- io.binaries) yield b.asUInt()).reduce(Cat(_, _)).orR()
      for (i <- 0 until height)
        b_counter(i).io.en := io.binaries(i)
      write_reg := put
      when(put) {
        for (i <- 0 until height) {
          when(io.binaries(i)) {
            mask_reg(i) := r_counter.io.out
          }.otherwise {
            mask_reg(i) := no_pe.U
          }
        }
      }.otherwise {
        mask_reg := DontCare
      }

      val full = (for (i <- 0 until height)
        yield (io.binaries(i) & (b_counter(i).io.out === pe_n.U)).asUInt())
        .reduce(Cat(_, _))
        .orR()

      when(full) {
        for (i <- 0 until height) {
          b_counter(i).io.en := 0.B
          b_counter(i).io.clr := 1.B
        }
        r_counter.io.en := 0.B
        r_counter.io.clr := 1.B
        io.emit := 1.B
      }.otherwise {
        for (i <- 0 until height) {
          b_counter(i).io.clr := 0.B
        }
        r_counter.io.en := 1.B
        r_counter.io.clr := 0.B
        io.emit := 0.B
      }
    }
  }
}
