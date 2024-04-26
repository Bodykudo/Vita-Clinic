import { Module } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';

import { AppointmentsController } from './appointments.controller';

import { AppointmentsService } from './appointments.service';
import { PrismaService } from 'src/prisma.service';
import { BiomarkersService } from 'src/settings/biomarkers/biomarkers.service';
import { LaboratoryTestsService } from 'src/settings/laboratory-tests/laboratory-tests.service';
import { ModalitiesService } from 'src/settings/modalities/modalities.service';
import { ServicesService } from 'src/settings/services/services.service';
import { TherapiesService } from 'src/settings/therapies/therapies.service';

@Module({
  controllers: [AppointmentsController],
  providers: [
    AppointmentsService,
    JwtService,
    PrismaService,
    BiomarkersService,
    LaboratoryTestsService,
    ModalitiesService,
    ServicesService,
    TherapiesService,
  ],
})
export class AppointmentsModule {}
