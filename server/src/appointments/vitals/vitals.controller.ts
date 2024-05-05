import { Request } from 'express';
import {
  Controller,
  Post,
  Body,
  Get,
  Req,
  UnauthorizedException,
  UseGuards,
  Patch,
  Param,
  Delete,
  ValidationPipe,
} from '@nestjs/common';

import { VitalsService } from './vitals.service';
import { JwtGuard } from 'src/auth/guards/jwt.guard';

import { CreateVitalsDto, UpdateVitalsDto } from './dto/vitals.dto';
import type { Payload } from 'src/types/payload.type';

@Controller('vitals')
export class VitalsController {
  constructor(private readonly vitalsService: VitalsService) {}

  @UseGuards(JwtGuard)
  @Patch(':id')
  async update(
    @Param('id') id: string,
    @Body() updateVitalsDto: UpdateVitalsDto,
    @Req() request: Request,
  ) {
    const user: Payload = request['user'];

    if (user.role !== 'doctor') {
      throw new UnauthorizedException();
    }

    return this.vitalsService.update(id, updateVitalsDto);
  }
}
