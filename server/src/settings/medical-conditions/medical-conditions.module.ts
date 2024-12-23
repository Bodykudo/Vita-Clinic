import { Module } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';

import { MedicalConditionsController } from './medical-conditions.controller';

import { MedicalConditionsService } from './medical-conditions.service';
import { PrismaService } from 'src/prisma.service';
import { LogService } from 'src/log/log.service';

@Module({
  controllers: [MedicalConditionsController],
  providers: [MedicalConditionsService, JwtService, PrismaService, LogService],
})
export class MedicalConditionsModule {}
